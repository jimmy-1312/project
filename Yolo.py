import torch
from ultralytics import YOLO


class Yolo26():
    """
    Useful Functions:
        
        .change_scale_factor(scale_factor) --> to change the scale factor into your desire one
        
        .label(img_paths,save_paths) --> label the images in img_paths and save them in save_paths
        
        .boxes_cor(img_path) --> get the boundary box cordinates of every detected objects in single image

    """
    def __init__(self,scale_factor = 1):
        """
        Args:
            scale_factor: Integer of (1, 2, 3...), the larger of it, the better you can observe small object. 
                        The drawback is that the speed will be slower, normally 1 or 2 is pretty much enough.
        """

        self.scale_factor = scale_factor
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO("yolo26s.pt")
    
    def change_scale_factor(self,scale_factor):
        """
        Args:
            scale_factor: the value you want to change scale_factor to
        """
        self.scale_factor = scale_factor

    
    def detection(self,img_paths):
        """
        Args:
            img_paths: [path1,path2,...]
            A list of paths of images you want to predict
        """

        results = self.model.predict(source=img_paths, imgsz=640*self.scale_factor, device=self.device)
        return results

    def label(self,img_paths, save_paths):
        """
        Args:
            img_paths: [path1,path2,...]
                        A list of paths of images you want to predict

            save_paths: [save_path1,save_path2,...]
                        A list of save_path you want to save the output images
                        Note: save_path1 save the output image of path1, and go on
        """

        results = self.detection(img_paths)
        for i in range(len(results)):
            results[i].save(filename=save_paths[i])
    def boxes(self,img_path):
        """
        Args:
            img_path: path of image you want to get it's boxes cordinates of every detected objects
        Return:
            A boxes object which you can get imformation out of it
        """
        results = self.detection(img_path)
        return results[0].boxes
    


