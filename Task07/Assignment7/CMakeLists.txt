cmake_minimum_required(VERSION 3.10)
project(RayTracing)

SET(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17 )
FIND_PACKAGE(Threads)
add_executable(RayTracing main.cpp Object.hpp Vector.cpp Vector.hpp Sphere.hpp global.hpp Triangle.hpp Scene.cpp
        Scene.hpp Light.hpp AreaLight.hpp BVH.cpp BVH.hpp Bounds3.hpp Ray.hpp Material.hpp Intersection.hpp
        Renderer.cpp Renderer.hpp)
TARGET_LINK_LIBRARIES(RayTracing ${CMAKE_THREAD_LIBS_INIT})
