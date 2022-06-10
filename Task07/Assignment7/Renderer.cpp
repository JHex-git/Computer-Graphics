//
// Created by goksu on 2/25/20.
//

#include <fstream>
#include "Scene.hpp"
#include "Renderer.hpp"
//<multi threads>
#include <thread>
#include <mutex>
// #include <condition_variable>
std::mutex mutex_ins;
// #define MAX_QUEUE_SIZE 1000
// #define MAX_THREAD_SIZE 4

// std::mutex _lock;
// std::mutex _buf_lock;
// std::condition_variable _no_empty;
// std::condition_variable _no_full;
//</multi threads>

inline float deg2rad(const float& deg) { return deg * M_PI / 180.0; }

const float EPSILON = 0.01;

// The main render function. This where we iterate over all pixels in the image,
// generate primary rays and cast these rays into the scene. The content of the
// framebuffer is saved to a file.
void Renderer::Render(const Scene& scene)
{
//<multi threads>
    // exitThread = false;
    // std::vector<std::thread *> threads;
    // for (int i = 0; i < MAX_THREAD_SIZE; i++)
    // {
    //     std::thread *t = new std::thread([=] {
    //         Run(i);
    //     });

    //     threads.push_back(t);
    // }
    // task_id = 0;
//</multi threads>

    std::vector<Vector3f> framebuffer(scene.width * scene.height);

    float scale = tan(deg2rad(scene.fov * 0.5));
    float imageAspectRatio = scene.width / (float)scene.height;
    Vector3f eye_pos(278, 273, -800);
    int m = 0;


    // change the spp value to change sample ammount
    int spp = 256;
    std::cout << "SPP: " << spp << "\n";
    // for (uint32_t j = 0; j < scene.height; ++j) {
    //     for (uint32_t i = 0; i < scene.width; ++i) {
    //         // generate primary ray direction
    //         float x = (2 * (i + 0.5) / (float)scene.width - 1) *
    //                   imageAspectRatio * scale;
    //         float y = (1 - 2 * (j + 0.5) / (float)scene.height) * scale;

    //         Vector3f dir = normalize(Vector3f(-x, y, 1));
    //         for (int k = 0; k < spp; k++){
    //             framebuffer[m] += scene.castRay(Ray(eye_pos, dir), 0) / spp; 
    //             // <multi threads>
    //             // task_data d = {m, eye_pos, dir, 0, spp};
    //             // auto func = std::bind(&Renderer::CastRayTask, this, std::placeholders::_1);
    //             // task t = {d, func};
    //             // AddTask(t);
    //             // </multi threads>
    //         }
    //         m++;
    //     }
    //     UpdateProgress(j / (float)scene.height);
    // }
    //<multi threads>
    // exitThread = true;
    // for (auto &t : threads)
    // {
    //     t->join();
    // }
    //</multi threads>
    int process = 0;

	// 创造匿名函数，为不同线程划分不同块
	auto castRayMultiThreading = [&](uint32_t rowStart, uint32_t rowEnd, uint32_t colStart, uint32_t colEnd)
	{
		for (uint32_t j = rowStart; j < rowEnd; ++j) {
			int m = j * scene.width + colStart;
			for (uint32_t i = colStart; i < colEnd; ++i) {
				// generate primary ray direction
				float x = (2 * (i + 0.5) / (float)scene.width - 1) *
					imageAspectRatio * scale;
				float y = (1 - 2 * (j + 0.5) / (float)scene.height) * scale;

				Vector3f dir = normalize(Vector3f(-x, y, 1));
				for (int k = 0; k < spp; k++) {
					framebuffer[m] += scene.castRay(Ray(eye_pos, dir), 0) / spp;
				}
				m++;
				process++;
			}

			// 互斥锁，用于打印处理进程
			std::lock_guard<std::mutex> g1(mutex_ins);
			UpdateProgress(1.0*process / scene.width / scene.height);
		}
	};

	int id = 0;
	constexpr int bx = 5;
	constexpr int by = 5;
	std::thread th[bx * by];

	int strideX = (scene.width + 1) / bx;
	int strideY = (scene.height + 1) / by;

	// 分块计算光线追踪
	for (int i = 0; i < scene.height; i += strideX)
	{
		for (int j = 0; j < scene.width; j += strideY)
		{
			th[id] = std::thread(castRayMultiThreading, i, std::min(i + strideX, scene.height), j, std::min(j + strideY, scene.width));
			id++;
		}
	}

	for (int i = 0; i < bx*by; i++) th[i].join();
	UpdateProgress(1.f);

    // save framebuffer to file
    FILE* fp = fopen("binary.ppm", "wb");
    (void)fprintf(fp, "P6\n%d %d\n255\n", scene.width, scene.height);
    for (auto i = 0; i < scene.height * scene.width; ++i) {
        static unsigned char color[3];
        color[0] = (unsigned char)(255 * std::pow(clamp(0, 1, framebuffer[i].x), 0.6f));
        color[1] = (unsigned char)(255 * std::pow(clamp(0, 1, framebuffer[i].y), 0.6f));
        color[2] = (unsigned char)(255 * std::pow(clamp(0, 1, framebuffer[i].z), 0.6f));
        fwrite(color, 1, 3, fp);
    }
    fclose(fp);    
}
//<multi threads>
// void Renderer::AddTask(const task &t)
// {
//     while (true)
//     {
//         std::unique_lock l(_lock);
//         if (taskQueue.size() < MAX_QUEUE_SIZE)
//         {
//             taskQueue.push(t);
//             l.unlock();
//             _no_empty.notify_all();
//             break;
//         }
//         else
//         {
//             _no_full.wait(l);
//         }
//     }
// }

// void Renderer::Run(int id)
// {
//     while (true)
//     {
//         std::unique_lock l(_lock);
//         if (!taskQueue.empty())
//         {
//             task t = taskQueue.front();
//             taskQueue.pop();
//             l.unlock();

//             t.data.id = id;
//             t.func(t.data);
//             _no_full.notify_all();
//         }
//         else if(exitThread)
//         {
//             break;
//         }
//         else
//         {
//             _no_empty.wait_for(l, std::chrono::milliseconds(50));
//         }
//     }
// }

// void Renderer::CastRayTask(task_data data)
// {
//     auto c = renderer_scene->castRay(Ray(data.eye_pos, data.dir), data.depth) / data.ssp;

//     std::unique_lock l(_buf_lock);
//     framebuffer[data.m] += c;
//     int line = task_id / (renderer_scene->width * data.ssp);
//     task_id++;
//     UpdateProgress(line / renderer_scene->height);
// }
//</multi threads>