

//��Ϊ�������kernelʱ��Ҫ����һ���߳���Ŀ��
//����ÿ���̶߳������һ�����������ֻ��Ҫʹ
//��get_global_id��ȡ�����߳�id�Ϳ��������
kernel void add(global const int* a, global int* b)
{
	int gid = get_global_id(0);
	b[gid] = a[gid] + a[gid];
}