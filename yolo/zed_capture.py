
import rclpy
from rclpy.node import Node, Subscription
import cv2
import datetime
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import os
import time
class ZEDCap(Node):
    """
    path: path to the file to be written
    fourcc: codec (MJPG) works fine
    
    """

    def __init__(self, fourcc, path):
        super().__init__('zed_capture')
        self.path = path
        self.fourcc = fourcc
        self.video_writer = None


        
        self.image_subscriber = self.create_subscription(Image, 
        '/zed/zed_node/rgb/image_rect_color', 
        self.write_image_callback, 
        10) 
        self.frame_count = 0
        self.bridge = CvBridge()
        self.image_subscriber


    def initialize_video_writer(self,data):
        img_height = data.height
        img_width = data.width
        self.video_writer = cv2.VideoWriter(
            self.path, cv2.CAP_FFMPEG, self.fourcc, 20, (img_width, img_height))
        if self.video_writer.isOpened():
            self.get_logger().info(f'Successfully opened video file {self.path}')
        else:
            self.get_logger().error(f'Failed to open video file {self.path}')
            raise Exception("Failed to open video file")

    def write_image_callback(self, data):
        """
        Subscriber that receives rectified images from the zed camera
        and writes it into a file using the videow writer
        """
        if self.video_writer is None:
            self.initialize_video_writer(data)


        try:
            current_frame = self.bridge.imgmsg_to_cv2(data, 'bgr8')

        except CvBridgeError as e:
            print(e)
        
        
        self.get_logger().info('writing frame...')
        # Check the aspect ratio of the current frame
        
        
        self.video_writer.write(current_frame)
        self.frame_count += 1
        
    def release_video_writer(self):
        """
        Release video writer after you are done with it
        """
        file_size = os.path.getsize(self.path)
        file_size_mb = file_size / (1024 * 1024)
        self.video_writer.release()
        time.sleep(5)
        self.get_logger().info(f"Video writer released. {self.frame_count} frames written")
        self.get_logger().info(f"to {self.path}")
        self.get_logger().info(f"File size: {file_size_mb:.2f} MB")        

def main(args=None):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    home = os.path.expanduser('~')
    path = os.path.join(home, 'Desktop', 'captures')
    if not os.path.exists(path):
        os.mkdir(path)
    file_name = 'capture-' + datetime.datetime.now().strftime('%H-%M-%d-%m') + '.mp4'
    file_path = os.path.join(path, file_name)
    rclpy.init(args=args)
    try:
        zed_cap = ZEDCap(fourcc, file_path)
        rclpy.spin(zed_cap)
    
    except KeyboardInterrupt:
        print('Shutting down node..')
        zed_cap.release_video_writer()
        zed_cap.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

    

if __name__ == '__main__':
    print("Hello world!")
    main()
