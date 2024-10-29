#include "utils.h"
#include "infer.h"
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

class ros_node
{
public:
    ros::NodeHandle nh;

    std::string engine_file_path;
    std::string image_topic_out;
    std::shared_ptr<YoloDetector> detector;

    double timestamp{};

    int image_height{};
    int image_width{};

    ros::Publisher pub;
    ros_node();
    void callback_func(const sensor_msgs::ImageConstPtr& msg);
};

ros_node::ros_node()
{
    nh.param("image_height", image_height, 480);
    nh.param("image_width", image_width, 864);
    nh.param<std::string>("image_topic_out", image_topic_out, "/yolo/detect_result");
    nh.param<std::string>("engine_file_path", engine_file_path, "/home/cll/models/yolo11/yolo11x-864-480-fp32.engine");
    ROS_WARN("image_topic_out: %s", image_topic_out.c_str());
    pub = nh.advertise<sensor_msgs::Image>(image_topic_out, 10);

    //检查模型文件是否存在
    if (access(engine_file_path.c_str(), F_OK) == -1)
    {
        ROS_ERROR("Engine file does not exist!");
        exit(-1);
    }

    ROS_WARN("Load the engine file from %s", engine_file_path.c_str());
    pub = nh.advertise<sensor_msgs::Image>(image_topic_out, 10);
    detector = std::make_shared<YoloDetector>(engine_file_path);
}

void ros_node::callback_func(const sensor_msgs::ImageConstPtr& msg)
{
    //将ROS图像消息转换为OpenCV图像
    cv::Mat image_src = cv_bridge::toCvCopy(msg, "rgb8")->image;
    timestamp = msg->header.stamp.toSec();
    //848*480填充到864*480     -->     D455相机
    cv::Mat image_new;
    cv::copyMakeBorder(image_src, image_new, 0, 0, 8, 8, cv::BORDER_CONSTANT, 0);

    auto start = std::chrono::system_clock::now();
    std::vector<Detection> res = detector->inference(image_new);
    auto end = std::chrono::system_clock::now();

    auto tc = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) / 1000.;
    printf("cost %2.4lf ms\n", tc);

    //复制图像
    cv::Mat image_res;
    image_new.copyTo(image_res);
    YoloDetector::draw_image(image_res, res);

    std_msgs::Header header;
    header.frame_id = "yolo11_detect";
    header.stamp = ros::Time(timestamp);
    const sensor_msgs::ImagePtr msg_out = cv_bridge::CvImage(header, "rgb8", image_res).toImageMsg();
    pub.publish(msg_out);
}


int main(int argc, char* argv[])
{
    // cuda:0
    cudaSetDevice(0);
    ros::init(argc, argv, "yolo11_detect");
    ros::NodeHandle nh;
    std::string image_topic_in;
    nh.param<std::string>("image_topic_in", image_topic_in, "/camera/color/image_raw");
    auto detect = std::make_unique<ros_node>();
    //订阅输入图像话题
    ros::Subscriber sub = nh.subscribe(image_topic_in, 30, &ros_node::callback_func, detect.get(),
                                       ros::TransportHints().tcpNoDelay());

    ros::spin();
    return 0;
}
