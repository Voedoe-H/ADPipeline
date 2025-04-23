#include "externals/httplib.h"
#include "externals/json.hpp"
#include "externals/base64.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

cv::dnn::Net net; 

void handle_inference(const httplib::Request& req, httplib::Response& res)
{
    try
    {
        nlohmann::json body = nlohmann::json::parse(req.body);
        if (!body.contains("image"))
        {
            res.status = 400;
            res.set_content("Missing image field","text/plain");
            return;
        }

        std::string image_b64 = body["image"];
        std::string decoded_bytes = base64::from_base64(image_b64);
        std::cout << "Image received. Byte size: " << decoded_bytes.size() << std::endl;

        if (decoded_bytes.size() != 1024 * 1024)
        {
            res.status = 400;
            res.set_content("Wrong image size","text/plain");
        }

        cv::Mat img(1024,1024, CV_8UC1, (void*)decoded_bytes.data());
        cv::Mat inputBlob;
        img.convertTo(inputBlob, CV_32F, 1.0 / 255.0);
        inputBlob = inputBlob.reshape(1, {1, 1, 1024, 1024});

        net.setInput(inputBlob);
        cv::Mat output = net.forward();
        output = output.reshape(1, 1024);
        cv::Mat output_image;
        output.convertTo(output_image, CV_8U, 255.0);

        cv::imwrite("reconstructed.png", output_image);
        std::cout << "Saved output to reconstructed.png" << std::endl;
        //cv::imwrite("received_image.png", img);
        //std::cout << "Image saved as received_image.png" << std::endl;

        res.set_content("Works","text/html");
    }
    catch ( const std::exception& e)
    {
        res.status = 500;
        res.set_content("Formating Failed","text/html");
    }
}

int main()
{
    std::cout << "Loading model" << std::endl;
    try 
    {
        net = cv::dnn::readNetFromONNX("pytorchmodel_v2.onnx");
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to load model" << e.what() << std::endl;
        return -1;
    }
    
    std::cout << "Starting Server" << std::endl;
    httplib::Server server;

    server.Post("/infer", handle_inference);

    server.set_error_handler([](const auto& req, auto& res) 
    {
        auto fmt = "<p>Error Status: <span style='color:red;'>%d</span></p>";
        char buf[BUFSIZ];
        snprintf(buf, sizeof(buf), fmt, res.status);
        res.set_content(buf, "text/html");
    });

    server.listen("127.0.0.1",8080);
    
    return 0;
}