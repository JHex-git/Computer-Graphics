//
// Created by goksu on 2/25/20.
//
#include "Scene.hpp"
//<multi threads>
// #include <functional>
// #include <queue>
//</multi threads>
#pragma once
struct hit_payload
{
    float tNear;
    uint32_t index;
    Vector2f uv;
    Object* hit_obj;
};

//<multi thread>
// struct task_data
// {
//     int m;
//     Vector3f eye_pos;
//     Vector3f dir;
//     int depth;
//     int ssp;
//     int id;
// };


// struct task
// {
//     task_data data;
//     std::function<void(task_data)> func;
// };
//</multi threads>
class Renderer
{
public:
    void Render(const Scene& scene);
//<multi threads>
    // void Run(int id);
    // void CastRayTask(task_data);
    // void AddTask(const task&);
//</multi threads>
private:
//<multi threads>
    // std::queue<task> taskQueue;
    // bool exitThread;
    // std::vector<Vector3f> framebuffer;
    // Scene* renderer_scene;
    // int task_id;
//</multi threads>
};
