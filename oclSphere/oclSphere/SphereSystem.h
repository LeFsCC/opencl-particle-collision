
/*
    ��������ϵͳ�Ĳ�����ʼ��
    cpu �� gpu ֮�������(λ��)����
*/

#pragma once
#ifndef SPHERESYSTEM_H
#define SPHERESYSTEM_H

#include <cl/cl.h>

typedef unsigned int uint;
static size_t world_size = 64;

struct float3 {
    float x, y, z;
};

struct uint3 {
    unsigned int x, y, z;
};

// �������Ĳ����б�
typedef struct {
    float3 gravity;        // ����
    float global_damping;  // ȫ�ַ���ϵ��
    uint3 grid_size;       // ����ߴ�
    uint num_cells;        // ϸ������
    float3 world_origin;   // ����ԭ��
    float3 cell_size;      // ϸ����С
    float spring;          // ����ϵ��
    float damping;         // ����ϵ��
    float shear;           // ��Ӧ��ϵ��
    float attraction;      // ����
    float boundaryDamping; // �߽練��ϵ��
} sim_params;

// ������float����һ���ṹ����
static inline  float3 get_float_array(float x, float y, float z) {
    float3 t; 
    t.x = x; 
    t.y = y; 
    t.z = z; 
    return t;
}

class Spheres {

public:
    enum attr_type { POS, VEL};
	Spheres(uint, uint3);
    void update(float delta_time);   // ˢ�º���
    float* get_pos() { return cpu_pos; }   // �õ����ӵ�λ��
    void init_particle_params();    // ��ʼ������ϵͳ

private:
    void init_params();      // ��ʼ�������������
    void init_buffer();      // ��ʼ��С��İ뾶��λ�õȲ������������ڴ��gpu��
    float* get_gpu_data(attr_type array);   // ��gpu���õ�����
    void set_gpu_data(attr_type array, const float* data, int start, int count);  // �����ݴ���gpu��

    uint num_particles;
    float* cpu_pos;
    float* cpu_vel;

    cl_mem          gpu_pos;   // λ����Ϣ
    cl_mem          gpu_vel;   // �ٶ���Ϣ
    cl_mem     gpu_reod_pos;   // �����Ժ��λ��
    cl_mem     gpu_reod_vel;   // �����Ժ���ٶ�
    cl_mem         gpu_hash;   // ��ϣֵ
    cl_mem        gpu_index;   // ����ֵ
    cl_mem   gpu_cell_start;   // ϸ����ʼ��ַ
    cl_mem     gpu_cell_end;   // ϸ��������ַ

    sim_params params;         // gpu��Ҫ�õ��������������
    uint3 grid_size;           // ���Ӵ�С
    uint num_grid_cells;       // ϸ������
};

#endif SPHERESYSTEM_H