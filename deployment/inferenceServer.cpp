#include "externals/httplib.h"
#include "externals/json.hpp"
#include "externals/base64.hpp"
#include <iostream>


void handle_inference(const httplib::Request& req, httplib::Response& res)
{
    try
    {
        nlohmann::json body = nlohmann::json::parse(req.body);
        if (!body.contains("image")){
            res.status = 400;
            res.set_content("Missing image field","text/plain");
            return;
        }

        std::string image_b64 = body["image"];
        std::string decoded_bytes = base64::from_base64(image_b64);
        std::cout << "Image received. Byte size: " << decoded_bytes.size() << std::endl;

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
    std::cout << "Starting Server" << std::endl;
    httplib::Server server;

    server.Post("/infer", handle_inference);

    server.set_error_handler([](const auto& req, auto& res) {
        auto fmt = "<p>Error Status: <span style='color:red;'>%d</span></p>";
        char buf[BUFSIZ];
        snprintf(buf, sizeof(buf), fmt, res.status);
        res.set_content(buf, "text/html");
    });

    server.listen("127.0.0.1",8080);
    
    return 0;
}