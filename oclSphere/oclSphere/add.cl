

//因为运行这个kernel时需要设置一个线程数目，
//所以每个线程都会调用一次这个函数，只需要使
//用get_global_id获取它的线程id就可以求和了
kernel void add(global const int* a, global int* b)
{
	int gid = get_global_id(0);
	b[gid] = a[gid] + a[gid];
}