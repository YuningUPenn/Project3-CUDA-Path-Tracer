#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/device_ptr.h>
#include <iostream>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
static int* dev_matIDs;

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need
    cudaMalloc(&dev_matIDs, pixelcount * sizeof(int));
    cudaMemset(dev_matIDs, 0, pixelcount * sizeof(int));

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
    cudaFree(dev_matIDs);

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // TODO: implement antialiasing by jittering the ray
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
        );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
        }
    }
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
          // Set up the RNG
          // LOOK: this is how you use thrust's RNG! Please look at
          // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else {
                float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
                pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
                pathSegments[idx].color *= u01(rng); // apply some noise because why not
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
        }
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_intersections
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

        shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials
        );
        iterationComplete = true; // TODO: should be based off stream compaction results.

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
* Start with my own path trace
*/


// Helpers
__global__ void getMatIDs(int num_paths, ShadeableIntersection* intersections, int* matID) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) {
        return;
    }
    matID[idx] = intersections[idx].materialId;
}

struct IsActive {
    __device__ bool operator()(const PathSegment& p) const {
        return p.remainingBounces >= 0;
    }
};

__device__ glm::vec2 concentricSampleDisk(float u1, float u2) {
    float offsetX = 2.0f * u1 - 1.0f;
    float offsetY = 2.0f * u2 - 1.0f;
    if (offsetX == 0.0f && offsetY == 0.0f) {
        return glm::vec2(0.0f);
    }

    float r, theta;
    if (fabs(offsetX) > fabs(offsetY)) {
        r = offsetX;
        theta = (PI / 4.0f) * (offsetY / offsetX);
    }
    else {
        r = offsetY;
        theta = (PI / 2.0f) - (PI / 4.0f) * (offsetX / offsetY);
    }
    return glm::vec2(r * cosf(theta), r * sinf(theta));
}

__device__ float halton(int idx, int base) {
    float f = 1, r = 0;
    int i = idx;
    while (i > 0) {
        f *= 1.0f / base;
        r += f * (i % base);
        i /= base;
    }
    return r;
}
// Helpers END

__global__ void myGenerateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(0.0f, 0.0f, 0.0f);
        segment.throughput = glm::vec3(1.0f, 1.0f, 1.0f);

        // TODO: implement antialiasing by jittering the ray
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> jitter(0.0f, 1.0f);
        float deltax = jitter(rng);
        float deltay = jitter(rng);
        // do a stratified random sample with a 4 by 4 grid within a pixel, so length of 0.25 for one grid
        // do one for each iteration
        int group = index % 16; // 4 by 4 so 16 small boxes in a pixel
        deltax = deltax * 0.25f + (float)(group / 4) * 0.25f - 0.5f;
        deltay = deltay * 0.25f + (float)(group % 4) * 0.25f - 0.5f;

        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + deltax - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + deltay - (float)cam.resolution.y * 0.5f)
        );

        // Physically-based depth-of-field
        if (cam.lensRadius > 0.0f) {
            // halton tried, but not better than uniform as zig-zag increases
            //deltax = halton(index, 3);
            //deltay = halton(index, 3);
            glm::vec2 disk = concentricSampleDisk(jitter(rng), jitter(rng)) * cam.lensRadius;
            float ft = cam.focalDistance / glm::dot(segment.ray.direction, glm::normalize(cam.view));
            glm::vec3 focalPoint = cam.position + segment.ray.direction * ft;
            segment.ray.origin = cam.position + cam.right * disk.x + cam.up * disk.y;
            segment.ray.direction = glm::normalize(focalPoint - segment.ray.origin);
        }

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

__global__ void myShadeMaterial(
    int iter,
    int num_paths,
    int depth,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) {
        return;
    }

    // Handle all edge cases
    ShadeableIntersection intersection = shadeableIntersections[idx];
    if (pathSegments[idx].remainingBounces < 0) {
        return;
    }
    if (intersection.t < 0.0f) {
        pathSegments[idx].remainingBounces = -1;
        return;
    }
    Material material = materials[intersection.materialId];
    if (material.emittance > 0.0f) {
        pathSegments[idx].remainingBounces = -1;
        pathSegments[idx].color += pathSegments[idx].throughput * material.color * material.emittance;
        return;
    }
    
    glm::vec3 intersect = pathSegments[idx].ray.origin + pathSegments[idx].ray.direction * shadeableIntersections[idx].t;
    thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

    // Handle refract surface
    if (material.hasRefractive > 0.0f) {
        glm::vec3 norm = intersection.surfaceNormal;
        float cosi = glm::dot(norm, -pathSegments[idx].ray.direction);
        float etai = 1.0f;
        float etat = material.hasRefractive;
        if (cosi < 0.0f) { // this means the path is currently inside the object, so negate everything
            etai = material.hasRefractive;
            etat = 1.0f;
            cosi = -cosi;
            norm = -norm;
        }
        float eta = etai / etat;
        float k = 1.0f - eta * eta * (1.0f - cosi * cosi);
        // Fresnel Schlick Approximation
        float R0 = ((etai - etat) / (etai + etat));
        R0 = R0 * R0;
        float F = R0 + (1.0f - R0) * powf(1.0f - cosi, 5.0f);
        if (k < 0.0f) {
            // total reflect
            pathSegments[idx].ray.direction = glm::reflect(pathSegments[idx].ray.direction, shadeableIntersections[idx].surfaceNormal);
        }
        else {
            // partial reflect
            if (u01(rng) < F) {
                // reflect
                pathSegments[idx].ray.direction = glm::reflect(pathSegments[idx].ray.direction, shadeableIntersections[idx].surfaceNormal);
            }
            else {
                // refract
                pathSegments[idx].ray.direction = glm::normalize(pathSegments[idx].ray.direction * eta + norm * (eta * cosi - sqrtf(k)));
            }
        }

        // finalize refract
        pathSegments[idx].ray.origin = intersect + 0.0001f * pathSegments[idx].ray.direction;
        pathSegments[idx].throughput *= material.color;
        pathSegments[idx].remainingBounces -= 1;

        // Add Russian Roulette for refract
        float rr = fmaxf(fmaxf(pathSegments[idx].throughput.x, pathSegments[idx].throughput.y), pathSegments[idx].throughput.z);
        if (u01(rng) > rr) {
            pathSegments[idx].remainingBounces = -1;
            return;
        }
        pathSegments[idx].throughput /= rr;
        return;
    }

    // Handle reflect surface
    if (material.hasReflective > 0.0f) {
        pathSegments[idx].ray.direction = glm::reflect(pathSegments[idx].ray.direction, shadeableIntersections[idx].surfaceNormal);
        pathSegments[idx].ray.origin = intersect + 0.0001f * pathSegments[idx].ray.direction;
        pathSegments[idx].throughput *= material.color;
        pathSegments[idx].remainingBounces -= 1;
        
        // Add Russian Roulette for reflect surface
        float rr = fmaxf(fmaxf(pathSegments[idx].throughput.x, pathSegments[idx].throughput.y), pathSegments[idx].throughput.z);
        if (u01(rng) > rr) {
            pathSegments[idx].remainingBounces = -1;
            return;
        }
        pathSegments[idx].throughput /= rr;
        return;
    }
    
    // MIS STARTS HERE
    // Prepare the light and intersection point for nee
    int lightIdx;
    for (int i = 0; i < geoms_size; i++) {
        int matId = geoms[i].materialid;
        if (materials[matId].emittance > 0.0f) {
            lightIdx = i;
            break;
        }
    }

    // Get a random point on light
    glm::vec3 o = glm::vec3(-0.5f, -0.5f, -0.5f);
    glm::vec3 i = glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec3 j = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 k = glm::vec3(0.0f, 0.0f, 1.0f);
    o = multiplyMV(geoms[lightIdx].transform, glm::vec4(o, 1.0f)); // Position, so end with 1.0f
    i = multiplyMV(geoms[lightIdx].transform, glm::vec4(i, 0.0f)); // Directions, so end with 0.0f
    j = multiplyMV(geoms[lightIdx].transform, glm::vec4(j, 0.0f));
    k = multiplyMV(geoms[lightIdx].transform, glm::vec4(k, 0.0f));
    thrust::uniform_int_distribution<int> d(0, 5);
    int choose = d(rng);
    float area = -1.0f; // To use in future pdf calculation
    glm::vec3 lightNorm;
    float t1 = u01(rng);
    float t2 = u01(rng);
    glm::vec3 neePos = o;
    if (choose / 2 == 0) {
        neePos += t1 * i;
        neePos += t2 * j;
        area = glm::length(glm::cross(i, j));
        if (choose % 2 == 0) {
            lightNorm = multiplyMV(geoms[lightIdx].invTranspose, glm::vec4(-k, 0.0f));
        }
        else {
            lightNorm = multiplyMV(geoms[lightIdx].invTranspose, glm::vec4(k, 0.0f));
            neePos += k;
        }
        lightNorm = glm::normalize(lightNorm);
    }
    else if (choose / 2 == 1) {
        neePos += t1 * j;
        neePos += t2 * k;
        area = glm::length(glm::cross(j, k));
        if (choose % 2 == 0) {
            lightNorm = multiplyMV(geoms[lightIdx].invTranspose, glm::vec4(-i, 0.0f));
        }
        else {
            lightNorm = multiplyMV(geoms[lightIdx].invTranspose, glm::vec4(i, 0.0f));
            neePos += i;
        }
        lightNorm = glm::normalize(lightNorm);
    }
    else {
        neePos += t1 * k;
        neePos += t2 * i;
        area = glm::length(glm::cross(k, i));
        if (choose % 2 == 0) {
            lightNorm = multiplyMV(geoms[lightIdx].invTranspose, glm::vec4(-j, 0.0f));
        }
        else {
            lightNorm = multiplyMV(geoms[lightIdx].invTranspose, glm::vec4(j, 0.0f));
            neePos += j;
        }
        lightNorm = glm::normalize(lightNorm);
    }

    // Check Occlude
    Ray invNEERay;
    invNEERay.origin = intersect;
    invNEERay.direction = glm::normalize(neePos - intersect); // So it point from intersect point to nee sample point

    // Copy from computeIntersections
    // Copy START
    float t;
    glm::vec3 intersect_point;
    glm::vec3 normal;
    float t_min = FLT_MAX;
    int hit_geom_index = -1;
    bool outside = true;

    glm::vec3 tmp_intersect;
    glm::vec3 tmp_normal;

    for (int i = 0; i < geoms_size; i++)
    {
        Geom& geom = geoms[i];

        if (geom.type == CUBE)
        {
            t = boxIntersectionTest(geom, invNEERay, tmp_intersect, tmp_normal, outside);
        }
        else if (geom.type == SPHERE)
        {
            t = sphereIntersectionTest(geom, invNEERay, tmp_intersect, tmp_normal, outside);
        }

        if (t > 0.0f && t_min > t)
        {
            t_min = t;
            hit_geom_index = i;
            intersect_point = tmp_intersect;
            normal = tmp_normal;
        }
    }
    // Copy END
    // Check hit index only, if not light then occluded
    if (hit_geom_index == lightIdx) {
        float dist2 = glm::dot(neePos - intersect, neePos - intersect); // Square of the distance
        float cosX = glm::max(0.0f, glm::dot(shadeableIntersections[idx].surfaceNormal, invNEERay.direction)); // Cosine value for the angle at surface
        float cosY = glm::max(0.0f, glm::dot(lightNorm, -invNEERay.direction)); // Cosine value for the angle at light
        if (cosX > 0.0f && cosY > 0.0f) {
            float pdfA = 1.0f / area;
            float pL = pdfA * dist2 / cosY;
            float pB = cosX / PI;
            glm::vec3 Le = materials[geoms[lightIdx].materialid].emittance * materials[geoms[lightIdx].materialid].color;
            glm::vec3 f = materials[geoms[lightIdx].materialid].color / PI;
            glm::vec3 CL = f * Le * cosX / pL;
            float wL = pL / (pL + pB);
            pathSegments[idx].color += pathSegments[idx].throughput * CL * wL;
        }
    }
    // END MIS FOR NEE

    pathSegments[idx].throughput *= material.color;
    scatterRay(pathSegments[idx], intersect, shadeableIntersections[idx].surfaceNormal, materials[shadeableIntersections[idx].materialId], rng);

    // Add Russian Roulette for diffuse
    float rr = fmaxf(fmaxf(pathSegments[idx].throughput.x, pathSegments[idx].throughput.y), pathSegments[idx].throughput.z);
    if (u01(rng) > rr) {
        pathSegments[idx].remainingBounces = -1;
        return;
    }
    pathSegments[idx].throughput /= rr;
    return;
}

__global__ void myFinalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

void mypathtrace(uchar4* pbo, int frame, int iter) {
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    myGenerateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;
    
    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        //cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
        cudaMemset(dev_intersections, 0, num_paths * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_intersections
            );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

        
        // Sort to arrange paths and intersections by ID of materials
        getMatIDs << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_intersections, dev_matIDs);
        cudaDeviceSynchronize();
        thrust::device_ptr<int>  d_keys(dev_matIDs);
        thrust::device_ptr<PathSegment> d_paths(dev_paths);
        thrust::device_ptr<ShadeableIntersection> d_intersections(dev_intersections);
        auto values_begin = thrust::make_zip_iterator(thrust::make_tuple(d_paths, d_intersections));
        thrust::sort_by_key(d_keys, d_keys + num_paths, values_begin);
        

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.
        
        myShadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            num_paths,
            depth,
            dev_intersections,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_materials
            );
        cudaDeviceSynchronize();
        
        
        thrust::device_ptr<PathSegment> paths_begin(dev_paths);
        thrust::device_ptr<PathSegment> paths_end = paths_begin + num_paths;
        auto mid = thrust::partition(
            paths_begin,
            paths_end,
            IsActive()
        );
        cudaDeviceSynchronize();
        
        int num_active = mid - paths_begin;
        //num_paths = num_active;
        //std::cout << "paths: " << num_paths << "; active: " << num_active << std::endl;
        if (num_active <= 0) {
            iterationComplete = true;
        }
        

        if (depth > traceDepth) {
            iterationComplete = true;
        }
        
        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }
    
    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    myFinalGather << <numBlocksPixels, blockSize1d >> > (num_paths, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
