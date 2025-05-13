def infinite_data_loader(data_loader):
    '''
    将数据加载器（data_loader）转换为一个无限循环的迭代器。
    即：训练过程中数据会无限循环，而不会在数据集遍历完后停止。
    '''
    while True:
        for data in data_loader:
            yield data   # 逐个返回数据（类似 return，但不会中断循环）。