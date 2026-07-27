#pragma once

#include "categories/visual_perception/depth_estimation.h"
#include "categories/visual_perception/image_classification.h"
#include "categories/visual_perception/instance_perception.h"
#include "categories/visual_perception/semantic_segmentation.h"
#include "core/platform/async.h"
#include "image/image_io.h"
#include "pages/state.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace app {

struct VisionResult {
    std::string path;
    std::string summary;
    int width = 0;
    int height = 0;
};

class VisionService {
public:
    static core::async::Result<VisionResult> analyze(
        VisionTask task, const std::string& model_path, const std::string& input_path,
        const std::string& right_path, int backend, float score_threshold,
        float iou_threshold, std::atomic<float>* progress) {
        forge::media::Image image;
        std::string error;
        if (!forge::media::load_image(std::filesystem::u8path(input_path), image, error)) {
            return core::async::failure<VisionResult>("无法读取图片: " + error);
        }
        if (progress) progress->store(0.10f);
        const char* device = backend == kBackendCuda ? "CUDA0"
                           : backend == kBackendSycl ? "SYCL0" : "CPU";
        if (task == VisionTask::Classification) {
            return classify(model_path, image, device, progress);
        }
        if (task == VisionTask::SemanticSegmentation) {
            return semantic(model_path, image, device, progress);
        }
        if (task == VisionTask::MonocularDepth || task == VisionTask::StereoDepth) {
            forge::media::Image right;
            if (task == VisionTask::StereoDepth &&
                !forge::media::load_image(std::filesystem::u8path(right_path), right, error)) {
                return core::async::failure<VisionResult>("无法读取右视图: " + error);
            }
            return depth(task, model_path, image, right, device, progress);
        }
        return instances(task, model_path, image, device, score_threshold, iou_threshold, progress);
    }

private:
    static void pixel(std::vector<uint8_t>& data, int width, int height, int x, int y,
                      uint8_t r, uint8_t g, uint8_t b, float alpha = 1.0f) {
        if (x < 0 || y < 0 || x >= width || y >= height) return;
        uint8_t* p = data.data() + (static_cast<size_t>(y) * width + x) * 3;
        p[0] = static_cast<uint8_t>(p[0] * (1.0f - alpha) + r * alpha);
        p[1] = static_cast<uint8_t>(p[1] * (1.0f - alpha) + g * alpha);
        p[2] = static_cast<uint8_t>(p[2] * (1.0f - alpha) + b * alpha);
    }

    static void line(std::vector<uint8_t>& data, int width, int height,
                     float ax, float ay, float bx, float by,
                     uint8_t r, uint8_t g, uint8_t b, int thickness = 3) {
        const float dx = bx - ax, dy = by - ay;
        const int steps = std::max(1, static_cast<int>(std::max(std::abs(dx), std::abs(dy))));
        for (int i = 0; i <= steps; ++i) {
            const float t = static_cast<float>(i) / steps;
            const int x = static_cast<int>(std::round(ax + dx * t));
            const int y = static_cast<int>(std::round(ay + dy * t));
            for (int oy = -thickness / 2; oy <= thickness / 2; ++oy)
                for (int ox = -thickness / 2; ox <= thickness / 2; ++ox)
                    pixel(data, width, height, x + ox, y + oy, r, g, b);
        }
    }

    static void circle(std::vector<uint8_t>& data, int width, int height,
                       float cx, float cy, int radius, uint8_t r, uint8_t g, uint8_t b) {
        for (int y = -radius; y <= radius; ++y)
            for (int x = -radius; x <= radius; ++x)
                if (x * x + y * y <= radius * radius)
                    pixel(data, width, height, static_cast<int>(cx) + x,
                          static_cast<int>(cy) + y, r, g, b);
    }

    static core::async::Result<VisionResult> save(
        std::vector<uint8_t>& pixels, uint32_t width, uint32_t height,
        std::string summary, const char* prefix) {
        VisionResult value;
        value.path = outputPath(std::string(prefix) + "-" + timestamp() + ".png");
        value.summary = std::move(summary);
        value.width = static_cast<int>(width);
        value.height = static_cast<int>(height);
        std::string error;
        if (!forge::media::save_png(std::filesystem::u8path(value.path), width, height, 3,
                                    pixels.data(), error)) {
            return core::async::failure<VisionResult>("无法保存分析结果: " + error);
        }
        return core::async::success(std::move(value));
    }

    static core::async::Result<VisionResult> classify(
        const std::string& model_path, const forge::media::Image& image,
        const char* device, std::atomic<float>* progress) {
        auto params = classification_runtime_default_params(); params.device = device;
        auto runtime = classification_runtime_create(params);
        auto model = runtime ? classification_load_model(runtime, model_path.c_str()) : nullptr;
        auto session = model ? classification_create_session(model) : nullptr;
        if (!session) {
            classification_free_session(session); classification_free_model(model);
            classification_runtime_free(runtime);
            return core::async::failure<VisionResult>("图像分类模型加载失败");
        }
        if (progress) progress->store(0.45f);
        classification_result result{};
        const classification_image input{image.width, image.height, 3, image.pixels.data()};
        const bool ok = classification_classify(
            session, &input, classification_request_default_params(), &result);
        std::ostringstream text;
        if (ok) for (size_t i = 0; i < result.score_count; ++i) {
            const char* label = classification_model_get_label(model, result.scores[i].class_id);
            if (i) text << "  ·  ";
            text << (label ? label : "class") << ' ' << static_cast<int>(result.scores[i].score * 100) << '%';
        }
        classification_free_result(&result); classification_free_session(session);
        classification_free_model(model); classification_runtime_free(runtime);
        if (!ok) return core::async::failure<VisionResult>("图像分类失败");
        if (progress) progress->store(1.0f);
        auto pixels = image.pixels;
        return save(pixels, image.width, image.height, text.str(), "classification");
    }

    static core::async::Result<VisionResult> instances(
        VisionTask task, const std::string& model_path, const forge::media::Image& image,
        const char* device, float score, float iou, std::atomic<float>* progress) {
        auto params = instance_runtime_default_params(); params.device = device;
        auto runtime = instance_runtime_create(params);
        auto model = runtime ? instance_load_model(runtime, model_path.c_str()) : nullptr;
        auto session = model ? instance_create_session(model) : nullptr;
        if (!session) {
            instance_free_session(session); instance_free_model(model); instance_runtime_free(runtime);
            return core::async::failure<VisionResult>("视觉模型加载失败");
        }
        if (progress) progress->store(0.45f);
        auto request = instance_request_default_params();
        request.score_threshold = score; request.iou_threshold = iou;
        request.task = task == VisionTask::InstanceSegmentation ? INSTANCE_TASK_MASKS
                     : task == VisionTask::Pose ? INSTANCE_TASK_KEYPOINTS
                     : task == VisionTask::Obb ? INSTANCE_TASK_ORIENTED_BOXES
                                               : INSTANCE_TASK_BOXES;
        instance_result result{};
        const instance_image input{image.width, image.height, 3, image.pixels.data()};
        const bool ok = instance_perceive(session, &input, request, &result);
        auto pixels = image.pixels;
        static constexpr uint8_t colors[][3] = {
            {56, 189, 248}, {167, 139, 250}, {52, 211, 153}, {251, 146, 60}, {244, 114, 182}};
        static constexpr int bones[][2] = {{5,7},{7,9},{6,8},{8,10},{5,6},{5,11},{6,12},
                                           {11,12},{11,13},{13,15},{12,14},{14,16},{0,1},{0,2},{1,3},{2,4}};
        if (ok) for (size_t i = 0; i < result.instance_count; ++i) {
            const auto& item = result.instances[i]; const auto& color = colors[i % 5];
            const float x1 = item.x - item.width * 0.5f, y1 = item.y - item.height * 0.5f;
            const float x2 = item.x + item.width * 0.5f, y2 = item.y + item.height * 0.5f;
            if (item.mask) {
                const int left = std::max(0, static_cast<int>(x1)), top = std::max(0, static_cast<int>(y1));
                const int right = std::min(static_cast<int>(image.width), static_cast<int>(x2));
                const int bottom = std::min(static_cast<int>(image.height), static_cast<int>(y2));
                for (int y = top; y < bottom; ++y) for (int x = left; x < right; ++x) {
                    const int mx = std::clamp(static_cast<int>((x - x1) / item.width * item.mask_width), 0,
                                              static_cast<int>(item.mask_width) - 1);
                    const int my = std::clamp(static_cast<int>((y - y1) / item.height * item.mask_height), 0,
                                              static_cast<int>(item.mask_height) - 1);
                    if (item.mask[static_cast<size_t>(my) * item.mask_width + mx])
                        pixel(pixels, image.width, image.height, x, y, color[0], color[1], color[2], 0.38f);
                }
            }
            if (request.task == INSTANCE_TASK_ORIENTED_BOXES) {
                const float c = std::cos(item.angle), s = std::sin(item.angle);
                float px[4], py[4];
                const float dx[4] = {-item.width/2, item.width/2, item.width/2, -item.width/2};
                const float dy[4] = {-item.height/2, -item.height/2, item.height/2, item.height/2};
                for (int p = 0; p < 4; ++p) { px[p] = item.x + dx[p]*c - dy[p]*s; py[p] = item.y + dx[p]*s + dy[p]*c; }
                for (int p = 0; p < 4; ++p) line(pixels, image.width, image.height, px[p], py[p], px[(p+1)%4], py[(p+1)%4], color[0], color[1], color[2]);
            } else {
                line(pixels,image.width,image.height,x1,y1,x2,y1,color[0],color[1],color[2]);
                line(pixels,image.width,image.height,x2,y1,x2,y2,color[0],color[1],color[2]);
                line(pixels,image.width,image.height,x2,y2,x1,y2,color[0],color[1],color[2]);
                line(pixels,image.width,image.height,x1,y2,x1,y1,color[0],color[1],color[2]);
            }
            if (item.keypoints) {
                for (const auto& bone : bones) if (bone[0] < static_cast<int>(item.keypoint_count) && bone[1] < static_cast<int>(item.keypoint_count)) {
                    const auto& a = item.keypoints[bone[0]]; const auto& b = item.keypoints[bone[1]];
                    if (a.score > 0.25f && b.score > 0.25f) line(pixels,image.width,image.height,a.x,a.y,b.x,b.y,color[0],color[1],color[2],2);
                }
                for (size_t p = 0; p < item.keypoint_count; ++p) if (item.keypoints[p].score > 0.25f)
                    circle(pixels,image.width,image.height,item.keypoints[p].x,item.keypoints[p].y,4,255,255,255);
            }
        }
        const size_t count = result.instance_count;
        instance_free_result(&result); instance_free_session(session);
        instance_free_model(model); instance_runtime_free(runtime);
        if (!ok) return core::async::failure<VisionResult>("视觉分析失败或模型不支持所选任务");
        if (progress) progress->store(1.0f);
        return save(pixels, image.width, image.height, std::to_string(count) + " 个目标", "vision");
    }

    static core::async::Result<VisionResult> semantic(
        const std::string& model_path, const forge::media::Image& image,
        const char* device, std::atomic<float>* progress) {
        auto params = segmentation_runtime_default_params(); params.device = device;
        auto runtime = segmentation_runtime_create(params);
        auto model = runtime ? segmentation_load_model(runtime, model_path.c_str()) : nullptr;
        auto session = model ? segmentation_create_session(model) : nullptr;
        if (!session) {
            segmentation_free_session(session); segmentation_free_model(model); segmentation_runtime_free(runtime);
            return core::async::failure<VisionResult>("语义分割模型加载失败");
        }
        if (progress) progress->store(0.45f);
        segmentation_result result{};
        const segmentation_image input{image.width,image.height,3,image.pixels.data()};
        const bool ok = segmentation_segment(session,&input,segmentation_request_default_params(),&result);
        auto pixels = image.pixels;
        if (ok) for (size_t i = 0; i < static_cast<size_t>(result.width) * result.height; ++i) {
            uint8_t color[3]{}; const int id = result.class_map[i];
            if (!segmentation_model_get_color(model,id,color)) {
                color[0]=(id*67)&255; color[1]=(id*149)&255; color[2]=(id*211)&255;
            }
            pixel(pixels,image.width,image.height,static_cast<int>(i%result.width),static_cast<int>(i/result.width),color[0],color[1],color[2],0.48f);
        }
        segmentation_free_result(&result); segmentation_free_session(session);
        segmentation_free_model(model); segmentation_runtime_free(runtime);
        if (!ok) return core::async::failure<VisionResult>("语义分割失败");
        if (progress) progress->store(1.0f);
        return save(pixels,image.width,image.height,"语义分割完成","segmentation");
    }

    static core::async::Result<VisionResult> depth(
        VisionTask task, const std::string& model_path, const forge::media::Image& left,
        const forge::media::Image& right, const char* device, std::atomic<float>* progress) {
        auto params = depth_runtime_default_params(); params.device = device;
        auto runtime = depth_runtime_create(params);
        auto model = runtime ? depth_load_model(runtime,model_path.c_str()) : nullptr;
        auto session = model ? depth_create_session(model) : nullptr;
        if (!session) {
            depth_free_session(session); depth_free_model(model); depth_runtime_free(runtime);
            return core::async::failure<VisionResult>("深度模型加载失败");
        }
        if (progress) progress->store(0.45f);
        const depth_image a{left.width,left.height,3,left.pixels.data()};
        const depth_image b{right.width,right.height,3,right.pixels.data()};
        auto request = depth_request_default_params();
        request.task = task == VisionTask::StereoDepth ? DEPTH_TASK_STEREO : DEPTH_TASK_MONOCULAR;
        depth_map map{}; const bool ok = depth_estimate(session,&a,request.task==DEPTH_TASK_STEREO?&b:nullptr,request,&map);
        std::vector<uint8_t> pixels(static_cast<size_t>(map.width)*map.height*3);
        float lo=std::numeric_limits<float>::infinity(), hi=-lo;
        if (ok) for (size_t i=0;i<static_cast<size_t>(map.width)*map.height;++i) if(std::isfinite(map.data[i])) {lo=std::min(lo,map.data[i]);hi=std::max(hi,map.data[i]);}
        const float range=hi>lo?hi-lo:1.0f;
        if (ok) for(size_t i=0;i<static_cast<size_t>(map.width)*map.height;++i) {
            float t=std::clamp((map.data[i]-lo)/range,0.0f,1.0f);
            pixels[i*3]=static_cast<uint8_t>(255*std::clamp(1.5f-std::abs(4*t-3),0.0f,1.0f));
            pixels[i*3+1]=static_cast<uint8_t>(255*std::clamp(1.5f-std::abs(4*t-2),0.0f,1.0f));
            pixels[i*3+2]=static_cast<uint8_t>(255*std::clamp(1.5f-std::abs(4*t-1),0.0f,1.0f));
        }
        const uint32_t width=map.width,height=map.height; const auto kind=map.kind;
        depth_free_map(&map); depth_free_session(session); depth_free_model(model); depth_runtime_free(runtime);
        if (!ok) return core::async::failure<VisionResult>("深度估计失败或输入不匹配");
        if (progress) progress->store(1.0f);
        return save(pixels,width,height,kind==DEPTH_MAP_DISPARITY?"双目视差图":"相对深度图","depth");
    }

    static std::string timestamp() {
        const auto now = std::chrono::system_clock::now().time_since_epoch();
        return std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(now).count());
    }
};

} // namespace app
