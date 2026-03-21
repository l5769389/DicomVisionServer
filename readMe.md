1. 前端发起/dicom/loadFolder，后端加载指定路径的 dicom，返回给前端对应的 seriesId 及影像基础信息。
2. 前端对指定 series 选择某个视图模式（Stack、MPR、3D）创建视口（/view/create），后端返回给前端该视口的 viewId，后续前端对视口的所有操作都携带该 viewId， 后端需要将 viewId、seriesId 绑定。
3. 前端获取某个视口的 size 信息，调用：/view/setSize，此时后端已经指定了影像信息和视口信息，开始向前端发送影像信息。


后端加载影像的逻辑：
1. 加载指定路径的 dicom，将 seriesId 与 dicom 信息绑定。绑定seriesId、dicom、viewId类功能应该独立，不要放到view中，避免view的功能太复杂
2. /view/create，后端将 viewId、seriesId 绑定。
3. /view/setSize，后端设置 viewport 信息。
4. 我希望viewport中 中的图像仿射变换逻辑可以独立封装。影像加载的时候可以缓存，而不是实时的读取本地文件

后端需要添加一些测试接口的代码，方便我测试接口是否正确。


API
1. dicom
  a. /dicom/loadFolder:
req : {
    folderPath: ''
}
res : {
    seriesId: ''
}

2. view
  a. /view/create
req: {
    seriesId: '',
    viewType: '',
}
res: {
    viewId: ''
}

  b. /view/setSize
req: {
    opType: 'setSize',
    size: {
        width: number,
        height: number,
    },
    viewId: string
}

3. socket-io:
  a. 前端发送图像操作请求
req : {
    viewId: '',
    opType: '',
    subOpType: '',
    actionType: '', 
    x: number,
    y: number,
    zoom: number,
    scroll : number,
    hor_flip: boolean,
    ver_flip: boolean,
}

  b. 后端响应前端图像
# image res
{
    slice_info: {
        current: 0,
        total: 10,
    },
    window_info: {
        ww: '',
        wl: ''
    },
    image: '',
    viewId: '',
}