#pragma once

#include "eui_neo.h"
#include "pages/actions.h"
#include "pages/state.h"
#include "pages/theme.h"
#include "services/model_catalog.h"

#include <algorithm>
#include <cstdio>

namespace app {

inline const char* visionTaskName(VisionTask task) {
    switch(task) {
        case VisionTask::Classification: return "图像分类";
        case VisionTask::Detection: return "目标检测";
        case VisionTask::InstanceSegmentation: return "实例分割";
        case VisionTask::Pose: return "姿态估计";
        case VisionTask::Obb: return "旋转框 OBB";
        case VisionTask::SemanticSegmentation: return "语义分割";
        case VisionTask::MonocularDepth: return "单目深度";
        case VisionTask::StereoDepth: return "双目深度";
    }
    return "视觉分析";
}

inline void visionTaskButton(eui::Ui& ui,const std::string& id,const char* label,
                             VisionTask task,float x,float y,float width) {
    const bool selected=state.vision_task==task;
    components::ButtonStyle style(studioTheme(),selected);
    style.normal=selected?kAccentSoft:kSurfaceInset; style.hover=kSurfaceRaised;
    style.text=selected?kAccent:kTextSecondary; style.border={1.0f,selected?kAccentEdge:kBorderSoft};
    style.radius=10.0f;
    components::button(ui,id).position(x,y).size(width,36.0f).text(label).fontSize(12.0f)
        .style(style).disabled(state.busy).onClick([task]{selectVisionTask(task);}).build();
}

inline void chooseVisionImage(bool right) {
    const std::string path=eui::platform::chooseFile({right?"选择右视图":"选择分析图片",
                                                       {".png",".jpg",".jpeg"}});
    if(path.empty()) return;
    if(right) state.vision_right_path=path; else state.vision_input_path=path;
    state.vision_output_path.clear(); state.vision_summary.clear();
    state.has_error=false; state.status="图片已就绪";
}

inline void composeVision(eui::Ui& ui,float x,float y,float width,float height) {
    const float inspector=std::clamp(width*0.29f,286.0f,350.0f);
    const float gap=16.0f, canvas_w=width-inspector-gap;
    panel(ui,"vision.canvas.panel",x,y,canvas_w,height);
    panel(ui,"vision.inspector.panel",x+canvas_w+gap,y,inspector,height);

    const float chip_gap=8.0f, chip_w=(canvas_w-48.0f-chip_gap*3.0f)/4.0f;
    const VisionTask tasks[8]={VisionTask::Classification,VisionTask::Detection,
        VisionTask::InstanceSegmentation,VisionTask::Pose,VisionTask::Obb,
        VisionTask::SemanticSegmentation,VisionTask::MonocularDepth,VisionTask::StereoDepth};
    for(int i=0;i<8;++i) visionTaskButton(ui,"vision.task."+std::to_string(i),visionTaskName(tasks[i]),tasks[i],
        x+24.0f+(i%4)*(chip_w+chip_gap),y+20.0f+(i/4)*44.0f,chip_w);

    const float media_y=y+116.0f, media_h=height-140.0f;
    const std::string shown=state.vision_output_path.empty()?state.vision_input_path:state.vision_output_path;
    if(shown.empty()) {
        ui.rect("vision.dropzone").position(x+24.0f,media_y).size(canvas_w-48.0f,media_h)
            .color(kSurfaceInset).radius(16.0f).border(1.0f,kBorder).onClick([]{chooseVisionImage(false);}).build();
        text(ui,"vision.empty.icon","＋",x+24.0f,media_y+media_h*0.34f,canvas_w-48.0f,48.0f,36.0f,kBlue,450);
        text(ui,"vision.empty.title","选择一张图片开始分析",x+24.0f,media_y+media_h*0.34f+54.0f,canvas_w-48.0f,28.0f,16.0f,kText,680);
        text(ui,"vision.empty.hint","支持 PNG 和 JPEG",x+24.0f,media_y+media_h*0.34f+86.0f,canvas_w-48.0f,20.0f,kFontCaption,kMuted,520);
    } else {
        components::image(ui,"vision.preview").position(x+24.0f,media_y).size(canvas_w-48.0f,media_h)
            .source(shown).contain().radius(14.0f).build();
        if(!state.vision_summary.empty()) {
            ui.rect("vision.result.badge").position(x+40.0f,media_y+media_h-48.0f).size(canvas_w-80.0f,34.0f)
                .color({0.035f,0.047f,0.070f,0.92f}).radius(10.0f).build();
            text(ui,"vision.result.summary",state.vision_summary,x+52.0f,media_y+media_h-41.0f,canvas_w-104.0f,20.0f,kFontCaption,kText,600);
        }
    }

    const float ix=x+canvas_w+gap+22.0f, iw=inspector-44.0f;
    sectionLabel(ui,"vision.input.label","输入",ix,y+22.0f,iw);
    components::button(ui,"vision.input.choose").position(ix,y+48.0f).size(iw,40.0f)
        .text(state.vision_input_path.empty()?"选择图片":"更换图片").icon(0xF07C).theme(studioTheme(),false)
        .radius(10.0f).disabled(state.busy).onClick([]{chooseVisionImage(false);}).build();
    text(ui,"vision.input.file",filenameOnly(state.vision_input_path),ix,y+94.0f,iw,18.0f,kFontOverline,kMuted,520);
    float next=y+126.0f;
    if(state.vision_task==VisionTask::StereoDepth) {
        components::button(ui,"vision.right.choose").position(ix,next).size(iw,40.0f)
            .text(state.vision_right_path.empty()?"选择右视图":"更换右视图").theme(studioTheme(),false)
            .radius(10.0f).disabled(state.busy).onClick([]{chooseVisionImage(true);}).build();
        text(ui,"vision.right.file",filenameOnly(state.vision_right_path),ix,next+46.0f,iw,18.0f,kFontOverline,kMuted,520);
        next+=80.0f;
    }
    if(state.vision_task==VisionTask::Detection||state.vision_task==VisionTask::InstanceSegmentation||
       state.vision_task==VisionTask::Pose||state.vision_task==VisionTask::Obb) {
        char score[32]{}; std::snprintf(score,sizeof(score),"置信度  %.0f%%",state.vision_score_threshold*100.0f);
        text(ui,"vision.score.label",score,ix,next,iw,20.0f,kFontCaption,kTextSecondary,620);
        ui.stack("vision.score.wrap").position(ix,next+24.0f).size(iw,24.0f).content([&]{
            components::slider(ui,"vision.score").size(iw,24.0f).value(state.vision_score_threshold)
                .theme(studioTheme()).onChange([](float v){state.vision_score_threshold=std::clamp(v,0.05f,0.95f);}).build();}).build();
        next+=66.0f;
        char iou[32]{}; std::snprintf(iou,sizeof(iou),"IOU  %.0f%%",state.vision_iou_threshold*100.0f);
        text(ui,"vision.iou.label",iou,ix,next,iw,20.0f,kFontCaption,kTextSecondary,620);
        ui.stack("vision.iou.wrap").position(ix,next+24.0f).size(iw,24.0f).content([&]{
            components::slider(ui,"vision.iou").size(iw,24.0f).value(state.vision_iou_threshold)
                .theme(studioTheme()).onChange([](float v){state.vision_iou_threshold=std::clamp(v,0.05f,0.95f);}).build();}).build();
    }
    const bool running=state.busy&&state.tool==Tool::Vision;
    components::button(ui,"vision.run").position(ix,y+height-64.0f).size(iw,44.0f)
        .text(running?"正在分析…":"开始分析").icon(0xF04B).theme(studioTheme(),true).radius(11.0f)
        .disabled(state.busy||state.vision_input_path.empty()).onClick(submitVision).build();
}

} // namespace app
