import rclpy
from rclpy.node import Node, Subscription
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import torch

class YOLO(Node): #node
    """
    
    This class is a subscriber that subscribes
    to a given topic (presumably that has message type sensor_msgs/Image)
    converts the message to a NumPy array using OpenCV
    and passes it through yolov5
    
    """



    def __init__(self, model): # constructor
        super().__init__('yolo')
        
        self.image_subscriber: Subscription = self.create_subscription( # from parent node
            Image,
            '/zed/zed_node/rgb/image_rect_color',
            self.display_image_callback, 
            10
        )
        self.depth_subscriber: Subscription = self.create_subscription(
            Image,
            '/zed/zed_node/depth/depth_registered',
            self.depth_callback,
            10
        )
        
        #defining 
        self.model = model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.classes = self.model.names
        self.bridge: CvBridge = CvBridge()
        self.image_subscriber
        self.depth_subscriber
        self.depth = None # depth in meters 
        self.get_logger().info(f"Using Device: {self.device}")

    def display_image_callback(self, data):
        
        
        self.get_logger().info(f"Receiving rectified images from ZED camera")
        current_frame = self.bridge.imgmsg_to_cv2(data)
        result = self.score_frame(current_frame)
        depth = self.get_depth()
        current_frame = self.plot_boxes(current_frame, result, depth) 

        cv2.imshow("Rectified Stereo", current_frame)

        cv2.waitKey(1)
    

    def depth_callback(self, data):
        """
        gets the depth array from the zed camera
        by converting it to a opencv image
        """
        data = self.bridge.imgmsg_to_cv2(data)
        self.depth = data

    def get_depth(self):
        """ 
        Depth getter
        """
        if self.depth is not None:
            return self.depth
        
        return None


    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    
    def get_depth_to_object(self, x1, y1, x2, y2, depth):
        """
        :param x1, x2, y1, y2: coordinates of the corners of the bounding box
        :returns the depth at the center of the bounding box

        """
        
        
        x_mid = x1 + (x2 - x1) // 2 
        y_mid = y1 + (y2 - y1) // 2
        
        if depth is not None:
            try:
                return x_mid, y_mid, depth[y_mid, x_mid]
            except IndexError:
                return None
        
        return None

        
        

    def infer(self, img):
        """
        Passes image through YOLO model
        """
        result = self.model(img)
        
        return result

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord
    

    def plot_boxes(self, frame, result, depth):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        
        labels, cord = result
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.3:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                depth_tuple = self.get_depth_to_object(x1, y1, x2, y2, depth)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, bgr, 2)
                if depth_tuple:
                    x_mid, y_mid, depth_to_object = depth_tuple
                    cv2.putText(frame, str(depth_to_object), (x1 + 80, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.circle(frame, (x_mid, y_mid), 5, (0, 0, 255), -1)

        return frame

    

def main(args=None):
    rclpy.logging.get_logger("Loading YOLO model!")
    model = torch.hub.load("ultralytics/yolov5", "yolov5l")
    rclpy.logging.get_logger("Model Loaded!")

    rclpy.init(args=args)

    yolo = YOLO(model)

    rclpy.spin(yolo)

    yolo.destroy_node()

    rclpy.shutdown

if __name__ == "__main__":
    main()
