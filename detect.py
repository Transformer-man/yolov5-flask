from utils.datasets import *
from utils.utils import *

def detect(source, out, weights):
    source, out, weights, imgsz = source, out, weights, 640
    # Initialize
    device = torch_utils.select_device('cpu')
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    # Load model
    google_utils.attempt_download(weights)
    model = torch.load(weights, map_location=device)['model']
    model.to(device).eval()
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz)
    # Get names and colors
    names = model.names if hasattr(model, 'names') else model.modules.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    # Run inference
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        t1 = time.time()
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, 0.4, 0.5,
                               fast=True, classes=None, agnostic=False)
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s
            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                for *xyxy, conf, cls in det:
                    # Add bbox to image
                    label = '%s%.2f' % (names[int(cls)], conf)
                    im0 = plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                    # xmin,ymin, xmax,ymax = int(xyxy[0]), int(xyxy[1]),int(xyxy[2]), int(xyxy[3])
                    # xcenter = xmin + (xmax - xmin) / 2
                    # ycenter = ymin + (ymax - ymin) / 2
                    # w = xmax - xmin
                    # h = ymax - ymin
            # Save results (image with detections)
            print('%sDone.  (%.3fs)' % (s, time.time() - t1))
            if dataset.mode == 'images':
                cv2.imwrite(save_path, im0)
            else:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                vid_writer.write(im0)

    print('Done. (%.3fs)' % (time.time() - t0))


source = './inference/inputs'
out = './inference/outputs'
weights = './weights/yolov5l.pt'

with torch.no_grad():
    detect(source, out, weights)
