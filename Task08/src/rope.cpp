#include <iostream>
#include <vector>

#include "CGL/vector2D.h"

#include "mass.h"
#include "rope.h"
#include "spring.h"

#define damp_factor 0.00005
#define isdamping true
namespace CGL {

    Rope::Rope(Vector2D start, Vector2D end, int num_nodes, float node_mass, float k, vector<int> pinned_nodes)
    {
        Vector2D nodes_distance = (end - start) / (num_nodes - 1);
        // TODO (Part 1): Create a rope starting at `start`, ending at `end`, and containing `num_nodes` nodes.
        for (int i = 0; i < num_nodes; ++i) {
            Mass *mass = new Mass(start + nodes_distance * i, node_mass, false);
            masses.push_back(mass);
        }
//        Comment-in this part when you implement the constructor
        for (auto &i : pinned_nodes) {
            masses[i]->pinned = true;
        }

        for (int i = 0; i < num_nodes - 1; ++i) {
            Spring *spring = new Spring(masses[i], masses[i + 1], k);
            springs.push_back(spring);
        }

    }

    void Rope::simulateEuler(float delta_t, Vector2D gravity)
    {
        for (auto &s : springs)
        {
            Vector2D f2to1 = -s->k * (s->m1->position - s->m2->position).unit() * ((s->m1->position - s->m2->position).norm() - s->rest_length);
            s->m1->forces += f2to1;
            s->m2->forces += -f2to1;
            // TODO (Part 2): Use Hooke's law to calculate the force on a node
        }

        for (auto &m : masses)
        {
            if (!m->pinned)
            {
                // TODO (Part 2): Add the force due to gravity, then compute the new velocity and position
                m->forces += m->mass * gravity;
                // TODO (Part 2): Add global damping
                Vector2D acceleration = m->forces / m->mass;
                m->velocity = m->velocity + acceleration * delta_t;

                m->position = m->position + m->velocity * delta_t;
            }

            // Reset all forces on each mass
            m->forces = Vector2D(0, 0);
        }
    }

    void Rope::simulateVerlet(float delta_t, Vector2D gravity)
    {
        for (auto &s : springs)
        {
            // Vector2D f2to1 = -s->k * (s->m1->position - s->m2->position).unit() * ((s->m1->position - s->m2->position).norm() - s->rest_length);
            // s->m1->forces += f2to1;
            // s->m2->forces += -f2to1;
            // TODO (Part 3): Simulate one timestep of the rope using explicit Verlet （solving constraints)
            if (s->m1->pinned && s->m2->pinned) continue;
            else if (s->m1->pinned && !s->m2->pinned) {
                s->m2->position = s->m1->position + (s->m2->position - s->m1->position).unit() * s->rest_length;
            }
            else if (!s->m1->pinned && s->m2->pinned) {
                s->m1->position = s->m2->position + (s->m1->position - s->m2->position).unit() * s->rest_length;
            }
            else {
                float half_distance = ((s->m1->position - s->m2->position).norm() - s->rest_length) / 2;
                s->m1->position += half_distance * (s->m2->position - s->m1->position).unit();
                s->m2->position += half_distance * (s->m1->position - s->m2->position).unit();
            }
        }

        for (auto &m : masses)
        {
            if (!m->pinned)
            {
                Vector2D temp_position = m->position;
                m->forces += m->mass * gravity;
                Vector2D acceleration = m->forces / m->mass;
                // TODO (Part 3.1): Set the new position of the rope mass
                if (isdamping) {
                    m->position = m->position + (1 - damp_factor) * (m->position - m->last_position) + acceleration * delta_t * delta_t;
                }
                else {
                    m->position = 2 * m->position - m->last_position + acceleration * delta_t * delta_t;
                }
                // TODO (Part 4): Add global Verlet damping
                m->last_position = temp_position;
            }
            m->forces = Vector2D(0, 0);
        }
    }
}
