
 ////////////////////////////////////////////////////////////////////////////////
 // Common definitions
 ////////////////////////////////////////////////////////////////////////////////
#define UMAD(a, b, c)  ( (a) * (b) + (c) )

typedef struct {
    float x, y, z;
} m_float3;

typedef struct {
    uint x, y, z;
}m_uint3;

typedef struct {
    m_float3 gravity;
    float global_damping;
    m_uint3 grid_size;
    uint num_cells;
    m_float3 world_origin;
    m_float3 cell_size;
    float spring;
    float damping;
    float shear;
    float attraction;
    float boundary_damping;
} sim_params;


// 初始化
__kernel void integrate(__global float4* gpu_pos,__global float4* gpu_vel,__constant sim_params* params,float delta_time,uint num_particles) {
    const uint index = get_global_id(0);
    if(index >= num_particles)
        return;

    float4 pos = gpu_pos[index];
    float4 vel = gpu_vel[index];

    //重力
    vel += (float4)(params->gravity.x, params->gravity.y, params->gravity.z, 0) * delta_time;
    vel *= params->global_damping;

    //新的位置
    pos += vel * delta_time;

    //与边界之间的碰撞
    if(pos.x < -1.0F + pos.w) {
        pos.x = -1.0F + pos.w;
        vel.x *= params->boundary_damping;
    }
    if(pos.x > 1.0F - pos.w) {
        pos.x = 1.0F - pos.w;
        vel.x *= params->boundary_damping;
    }

    if(pos.y < -1.0F + pos.w) {
        pos.y = -1.0F + pos.w;
        vel.y *= params->boundary_damping;
    }
    if(pos.y > 1.0F - pos.w) {
        pos.y = 1.0F - pos.w;
        vel.y *= params->boundary_damping;
    }

    if(pos.z < -1.0F + pos.w) {
        pos.z = -1.0F + pos.w;
        vel.z *= params->boundary_damping;
    }
    if(pos.z > 1.0F - pos.w) {
        pos.z = 1.0F - pos.w;
        vel.z *= params->boundary_damping;
    }

    //记录新的位置和速度
    gpu_pos[index] = pos;
    gpu_vel[index] = vel;
}


// 存粒子细胞的哈希值和索引
int4 get_grid_pos(float4 p, __constant sim_params* params) {
    int4 gridPos;
    gridPos.x = (int)floor((p.x - params->world_origin.x) / params->cell_size.x);
    gridPos.y = (int)floor((p.y - params->world_origin.y) / params->cell_size.y);
    gridPos.z = (int)floor((p.z - params->world_origin.z) / params->cell_size.z);
    gridPos.w = 0;
    return gridPos;
}

//  得到细胞的哈希值
uint get_grid_hash(int4 gridPos, __constant sim_params* params) {
    //Wrap addressing, assume power-of-two grid dimensions
    gridPos.x = gridPos.x & (params->grid_size.x - 1);
    gridPos.y = gridPos.y & (params->grid_size.y - 1);
    gridPos.z = gridPos.z & (params->grid_size.z - 1);
    return UMAD(UMAD(gridPos.z, params->grid_size.y, gridPos.y), params->grid_size.x, gridPos.x);
}

//Calculate grid hash value for each particle
__kernel void calc_hash(__global uint* d_Hash,__global uint* gpu_index,__global const float4* gpu_pos,__constant sim_params* params,uint num_particles) {
    const uint index = get_global_id(0);
    if(index >= num_particles)
        return;

    float4 p = gpu_pos[index];

    //得到细胞的地址
    int4  gridPos = get_grid_pos(p, params);
    uint gridHash = get_grid_hash(gridPos, params);

    //存储各自的哈希值和索引
    d_Hash[index] = gridHash;
    gpu_index[index] = index;
}

// 查找单元格边界，并通过排序索引重新排序位置和速度
__kernel void memset_gpu(__global uint* gpu_data,uint val,uint N) {
    if(get_global_id(0) < N)
        gpu_data[get_global_id(0)] = val;
}

__kernel void find_cell_bounds_reorder(
    __global uint* gpu_cell_start,
    __global uint* gpu_cell_end,
    __global float4* gpu_reorder_pos,
    __global float4* gpu_reorder_vel,
    __global const uint* d_Hash,
    __global const uint* gpu_index,
    __global const float4* gpu_pos,
    __global const float4* gpu_vel,
    __local uint* localHash,
    uint    num_particles) {
    uint hash;
    const uint index = get_global_id(0);

    if(index < num_particles) {
        hash = d_Hash[index];
        localHash[get_local_id(0) + 1] = hash;
        if(index > 0 && get_local_id(0) == 0)
            localHash[0] = d_Hash[index - 1];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if(index < num_particles) {
        if(index == 0)
            gpu_cell_start[hash] = 0;
        else{
            if(hash != localHash[get_local_id(0)])
                gpu_cell_end[localHash[get_local_id(0)]] = gpu_cell_start[hash] = index;
        };
        if(index == num_particles - 1)
            gpu_cell_end[hash] = num_particles;
        uint sortedIndex = gpu_index[index];
        float4 pos = gpu_pos[sortedIndex];
        float4 vel = gpu_vel[sortedIndex];

        gpu_reorder_pos[index] = pos;
        gpu_reorder_vel[index] = vel;
    }
}

// 处理碰撞(计算加速度)
float4 collideSpheres(float4 posA,float4 posB,float4 velA,float4 velB,float radiusA,float radiusB,
    float spring,float damping,float shear,float attraction) {

    //得到相对位置
    float4     rel_pos = (float4)(posB.x - posA.x, posB.y - posA.y, posB.z - posA.z, 0);
    float        dist = sqrt(rel_pos.x * rel_pos.x + rel_pos.y * rel_pos.y + rel_pos.z * rel_pos.z);
    float collde_dist = radiusA + radiusB;

    float4 force = (float4)(0, 0, 0, 0);
    if(dist < collde_dist) {
        float4 norm = (float4)(rel_pos.x / dist, rel_pos.y / dist, rel_pos.z / dist, 0);

        //相对速度
        float4 rel_vel = (float4)(velB.x - velA.x, velB.y - velA.y, velB.z - velA.z, 0);

        //剪应力带来的切向速度
        float relVelDotNorm = rel_vel.x * norm.x + rel_vel.y * norm.y + rel_vel.z * norm.z;
        float4 tanVel = (float4)(rel_vel.x - relVelDotNorm * norm.x, rel_vel.y - relVelDotNorm * norm.y, rel_vel.z - relVelDotNorm * norm.z, 0);

        //弹性力
        float spring_factor = -spring * (collde_dist - dist);
        force = (float4)(
            spring_factor * norm.x + damping * rel_vel.x + shear * tanVel.x + attraction * rel_pos.x,
            spring_factor * norm.y + damping * rel_vel.y + shear * tanVel.y + attraction * rel_pos.y,
            spring_factor * norm.z + damping * rel_vel.z + shear * tanVel.z + attraction * rel_pos.z,
            0);
    }
    return force;
}

__kernel void collide(
    __global float4* gpu_vel,
    __global const float4* gpu_reorder_pos,
    __global const float4* gpu_reorder_vel,
    __global const uint* gpu_index,
    __global const uint* gpu_cell_start,
    __global const uint* gpu_cell_end,
    __constant sim_params* params,
    uint    num_particles
) {
    uint index = get_global_id(0);
    if(index >= num_particles)
        return;

    float4   pos = gpu_reorder_pos[index];
    float4   vel = gpu_reorder_vel[index];
    float4 force = (float4)(0, 0, 0, 0);

    //得到网格的位置
    int4 gridPos = get_grid_pos(pos, params);

    //计算与周围cell的碰撞结果
    for(int z = -1; z <= 1; z++)
        for(int y = -1; y <= 1; y++)
        for(int x = -1; x <= 1; x++) {
        //得到该细胞中起始位置的例子hash
        uint   hash = get_grid_hash(gridPos + (int4)(x, y, z, 0), params);
        uint st = gpu_cell_start[hash];

        //跳过空cell
        if(st == 0xFFFFFFFFU)
            continue;

        //遍历这个cell中的粒子
        uint ed = gpu_cell_end[hash];
        for(uint j = st; j < ed; j++) {
            if(j == index)
                continue;

            float4 pos2 = gpu_reorder_pos[j];
            float4 vel2 = gpu_reorder_vel[j];

            //处理两个例子之间的碰撞
            force += collideSpheres( pos, pos2,vel, vel2,pos.w, pos2.w,params->spring, params->damping, params->shear, params->attraction);
        }
    }gpu_vel

    //得到新的位置
    [gpu_index[index]] = vel + force;
}

