inline void ComparatorPrivate(uint* keyA,uint* valA,uint* keyB,uint* valB,uint dir) {
    if ((*keyA > * keyB) == dir) {
        uint t;
        t = *keyA; *keyA = *keyB; *keyB = t;
        t = *valA; *valA = *valB; *valB = t;
    }
}

inline void ComparatorLocal(__local uint* keyA,__local uint* valA,__local uint* keyB,__local uint* valB,uint dir) {
    if ((*keyA > * keyB) == dir) {
        uint t;
        t = *keyA; *keyA = *keyB; *keyB = t;
        t = *valA; *valA = *valB; *valB = t;
    }
}

__kernel void merge_sort( __global uint* dst_key, __global uint* dst_val, __global uint* src_key, __global uint* src_val) {
    __local uint l_key[512U];
    __local uint l_val[512U];

    src_key += get_group_id(0) * 512U + get_local_id(0);
    src_val += get_group_id(0) * 512U + get_local_id(0);
    dst_key += get_group_id(0) * 512U + get_local_id(0);
    dst_val += get_group_id(0) * 512U + get_local_id(0);
    l_key[get_local_id(0)] = src_key[0];
    l_val[get_local_id(0)] = src_val[0];
    l_key[get_local_id(0) + (512U / 2)] = src_key[(512U / 2)];
    l_val[get_local_id(0) + (512U / 2)] = src_val[(512U / 2)];

    uint comparatorI = get_global_id(0) & ((512U / 2) - 1);

    for (uint size = 2; size < 512U; size <<= 1) {
        uint ddd = (comparatorI & (size / 2)) != 0;
        for (uint stride = size / 2; stride > 0; stride >>= 1) {
            barrier(CLK_LOCAL_MEM_FENCE);
            uint pos = 2 * get_local_id(0) - (get_local_id(0) & (stride - 1));
            ComparatorLocal(&l_key[pos + 0], &l_val[pos + 0],&l_key[pos + stride], &l_val[pos + stride],ddd);
        }
    }
    uint ddd = (get_group_id(0) & 1);
    for (uint stride = 512U / 2; stride > 0; stride >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        uint pos = 2 * get_local_id(0) - (get_local_id(0) & (stride - 1));
        ComparatorLocal(&l_key[pos + 0], &l_val[pos + 0], &l_key[pos + stride], &l_val[pos + stride],ddd);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    dst_key[0] = l_key[get_local_id(0) + 0];
    dst_val[0] = l_val[get_local_id(0) + 0];
    dst_key[(512U / 2)] = l_key[get_local_id(0) + (512U / 2)];
    dst_val[(512U / 2)] = l_val[get_local_id(0) + (512U / 2)];
}

__kernel void merge(__global uint* dst_key,__global uint* dst_val,__global uint* src_key,__global uint* src_val,
    uint arrayLength,uint size,uint stride,uint dir) {
    uint global_comparatorI = get_global_id(0);
    uint        comparatorI = global_comparatorI & (arrayLength / 2 - 1);

    uint ddd = dir ^ ((comparatorI & (size / 2)) != 0);
    uint pos = 2 * global_comparatorI - (global_comparatorI & (stride - 1));

    uint keyA = src_key[pos + 0];
    uint valA = src_val[pos + 0];
    uint keyB = src_key[pos + stride];
    uint valB = src_val[pos + stride];

    ComparatorPrivate(&keyA, &valA,&keyB, &valB,ddd);

    dst_key[pos + 0] = keyA;
    dst_val[pos + 0] = valA;
    dst_key[pos + stride] = keyB;
    dst_val[pos + stride] = valB;
}
