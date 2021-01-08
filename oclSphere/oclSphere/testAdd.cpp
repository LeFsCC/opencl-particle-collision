//#include<CL/cl.h>
//#include<iostream>
//using namespace std;
//#define _CRT_SECURE_NO_WARNINGS
//
//int main()
//{
//	cl_int err;
//	cl_uint counts;
//	cl_context context = NULL;
//	err = clGetPlatformIDs(0, 0, &counts);
//	if (err != CL_SUCCESS)puts("获取可用平台数量失败！"); else printf("获取可用平台数量成功！%d\n", counts);
//	cl_platform_id* platm = new cl_platform_id[counts];
//	err = clGetPlatformIDs(counts, platm, NULL);
//	if (err != CL_SUCCESS)puts("获取可用平台失败！"); else puts("获取可用平台成功！");
//	for (int i = 0; i < counts; i++)
//	{
//		size_t size;
//		err = clGetPlatformInfo(platm[i], CL_PLATFORM_NAME, 0, NULL, &size);
//		char* name = new char[size];
//		err = clGetPlatformInfo(platm[i], CL_PLATFORM_NAME, size, name, NULL);
//		cout << "GPU" << i + 1 << "名称：" << name << endl;
//		delete name;
//	}
//	cl_device_id cdi1;
//	clGetDeviceIDs(platm[1], CL_DEVICE_TYPE_GPU, 1, &cdi1, NULL);//查询平台上的设备
//	context = clCreateContext(NULL, 1, &cdi1, 0, 0, &err);//创建上下文
//	cl_command_queue ccq = clCreateCommandQueue(context, cdi1, 0, &err);//创建命令队列
//	FILE* fp = fopen("add.cl", "rb");
//	char t[50];
//	fseek(fp, 0, SEEK_END);
//	int fileset = ftell(fp);
//	fseek(fp, 0, SEEK_SET);
//	char ch; int i = 0; char* clcode = new char[fileset + 1];
//	fread(clcode, 1, fileset, fp);
//	clcode[fileset] = 0;
//	fclose(fp);
//	printf("%s\n", clcode);
//	cl_program cp = clCreateProgramWithSource(context, 1, (const char**)&clcode, NULL, &err);
//	err = clBuildProgram(cp, 0, 0, 0, 0, 0);
//	cl_kernel ck = clCreateKernel(cp, "add", &err);
//	int* a = new int[100], * b = new int[100];
//	for (int j = 0; j < 100; j++)
//		a[j] = j;
//	cl_mem mema = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(int) * 100, a, &err);
//	cl_mem memb = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * 100, NULL, &err);
//	err = clSetKernelArg(ck, 0, sizeof(cl_mem), &mema);
//	err = clSetKernelArg(ck, 1, sizeof(cl_mem), &memb);
//	size_t  gws[] = { 100 }, lws = { 1 };
//	err = clEnqueueNDRangeKernel(ccq, ck, 1, 0, (const size_t*)&gws, &lws, 0, 0, 0);
//	err = clEnqueueReadBuffer(ccq, memb, CL_TRUE, 0, sizeof(int) * 100, b, 0, 0, 0);
//	for (int j = 0; j < 100; j++)
//	{
//		j % 10 == 0 ? puts("") : 0;
//		printf("%04d ", a[j]);
//	}
//	puts("");
//	puts("GPU累加运算");
//	for (int j = 0; j < 100; j++)
//	{
//		j % 10 == 0 ? puts("") : 0;
//		printf("%04d ", b[j]);
//	}
//	getchar();
//	return 0;
//}
