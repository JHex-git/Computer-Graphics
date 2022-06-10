//
// Created by Göksu Güvendiren on 2019-05-14.
//

#include "Scene.hpp"


void Scene::buildBVH() {
    printf(" - Generating BVH...\n\n");
    this->bvh = new BVHAccel(objects, 1, BVHAccel::SplitMethod::NAIVE);
}

Intersection Scene::intersect(const Ray &ray) const
{
    return this->bvh->Intersect(ray);
}

//Randomly choose an incident light to pos from an object, 
//and change *pdf* to 1/A, A stands for the surface of the chosen object
void Scene::sampleLight(Intersection &pos, float &pdf) const
{
    float emit_area_sum = 0;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        if (objects[k]->hasEmit()){
            emit_area_sum += objects[k]->getArea();
        }
    }
    float p = get_random_float() * emit_area_sum;
    emit_area_sum = 0;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        if (objects[k]->hasEmit()){
            emit_area_sum += objects[k]->getArea();
            if (p <= emit_area_sum){
                objects[k]->Sample(pos, pdf);
                break;
            }
        }
    }
}

bool Scene::trace(
        const Ray &ray,
        const std::vector<Object*> &objects,
        float &tNear, uint32_t &index, Object **hitObject)
{
    *hitObject = nullptr;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        float tNearK = kInfinity;
        uint32_t indexK;
        Vector2f uvK;
        if (objects[k]->intersect(ray, tNearK, indexK) && tNearK < tNear) {
            *hitObject = objects[k];
            tNear = tNearK;
            index = indexK;
        }
    }


    return (*hitObject != nullptr);
}

// Implementation of Path Tracing
Vector3f Scene::castRay(const Ray &ray, int depth) const
{
    // TO DO Implement Path Tracing Algorithm here
    Vector3f lo(0.0f);
    
    Intersection shadingPoint;
    shadingPoint = intersect(ray);

    if (!shadingPoint.happened) {
        return lo;
    }

    if (shadingPoint.m->hasEmission()) {
        return shadingPoint.m->getEmission();
    }

    float pdf;
    Intersection lightPoint;
    sampleLight(lightPoint, pdf);
    //sampleLight Function will help to change the content of *intersection* and *pdf*
    
    Vector3f wi(normalize(lightPoint.coords - shadingPoint.coords));
    Ray testBlockRay(shadingPoint.coords, wi);
    Intersection block = intersect(testBlockRay);
    if (block.happened && (block.coords - lightPoint.coords).norm() < EPSILON) {//there is no block between light and shadingPoint
        // lo = lightPoint.emit * shadingPoint.m->eval(wi, -ray.direction, shadingPoint.normal) * dotProduct(shadingPoint.normal, wi) * dotProduct(lightPoint.normal, -wi)
        //     / dotProduct(lightPoint.coords - shadingPoint.coords, lightPoint.coords - shadingPoint.coords) / pdf;
        lo = lightPoint.emit * shadingPoint.m->eval(ray.direction, wi, shadingPoint.normal) * dotProduct(shadingPoint.normal, wi) * dotProduct(lightPoint.normal, -wi)
            / dotProduct(lightPoint.coords - shadingPoint.coords, lightPoint.coords - shadingPoint.coords) / pdf;
    }

    if (get_random_float() < RussianRoulette)
    {
        // Vector3f sample_wi = shadingPoint.m->sample(-ray.direction, shadingPoint.normal);
        Vector3f sample_wi = shadingPoint.m->sample(ray.direction, shadingPoint.normal).normalized();
        Ray sample_ray(shadingPoint.coords, sample_wi);
        Intersection sample_point = intersect(sample_ray);
        if (sample_point.happened && !sample_point.m->hasEmission())
        {
            // lo += castRay(sample_ray, depth + 1) * shadingPoint.m->eval(sample_wi, -ray.direction, shadingPoint.normal) * dotProduct(shadingPoint.normal, sample_wi)
            //     / RussianRoulette * 2 * M_PI;
            float objpdf = shadingPoint.m->pdf(ray.direction, sample_wi, shadingPoint.normal);
            lo += castRay(sample_ray, depth + 1) * shadingPoint.m->eval(ray.direction, sample_wi, shadingPoint.normal) * dotProduct(shadingPoint.normal, sample_wi)
                / RussianRoulette / objpdf;
        }
    }

    return lo;
}