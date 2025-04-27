#include "externals/httplib.h"
#include "externals/json.hpp"
#include "externals/base64.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

cv::dnn::Net net; 

double computeMSE(const cv::Mat& input, const cv::Mat& output) 
{
    cv::Mat diff;
    cv::absdiff(input, output, diff);
    diff = diff.mul(diff);
    return cv::mean(diff)[0];
}

double computeSSIM(const cv::Mat& img1, const cv::Mat& img2)
{
    const double C1 = 0.01 * 0.01;
    const double C2 = 0.03 * 0.03;

    cv::Mat I1, I2;
    img1.convertTo(I1, CV_32F);
    img2.convertTo(I2, CV_32F);

    cv::Mat I1_2 = I1.mul(I1);        // I1^2
    cv::Mat I2_2 = I2.mul(I2);        // I2^2
    cv::Mat I1_I2 = I1.mul(I2);       // I1 * I2

    cv::Mat mu1, mu2;
    cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);

    cv::Mat mu1_2 = mu1.mul(mu1);
    cv::Mat mu2_2 = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);

    cv::Mat sigma1_2, sigma2_2, sigma12;

    cv::GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;

    cv::GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;

    cv::GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    cv::Mat t1 = 2 * mu1_mu2 + C1;
    cv::Mat t2 = 2 * sigma12 + C2;
    cv::Mat t3 = t1.mul(t2);          // numerator

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);                  // denominator

    cv::Mat ssim_map;
    cv::divide(t3, t1, ssim_map);
    return cv::mean(ssim_map)[0];    
}

double compute_reconstruction_err(const cv::Mat& inp, const cv::Mat& out, double alpha = 0.84)
{
    double mse = computeMSE(inp,out);
    double ssim = computeSSIM(inp,out);
    double ssim_error = 1.0 - ssim;
    return (1-alpha) * mse + alpha * ssim_error;
}

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

        cv::Mat input_float, output_float;
        img.convertTo(input_float , CV_32F, 1.0 / 255.0);
        output.convertTo(output_float, CV_32F, 1.0 / 255.0);

        double error = compute_reconstruction_err(input_float, output_float);
        bool is_anomaly = error > 0.0595;

        nlohmann::json resp;
        resp["Anomaly"] = is_anomaly;
        res.set_content(resp.dump(),"text/html");
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

    server.listen("0.0.0.0",8080);
    
    return 0;
}