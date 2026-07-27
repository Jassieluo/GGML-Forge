#pragma once

#include "eui_neo.h"
#include "pages/state.h"
#include "pages/theme.h"

#include <algorithm>

namespace app {

inline void homeCard(eui::Ui& ui, const std::string& id, float x, float y, float width,
                     float height, const std::string& eyebrow, const std::string& title,
                     const std::string& description, const std::string& mark, Tool tool,
                     const eui::Color& accent) {
    ui.rect(id + ".bg").position(x,y).size(width,height).color(kSurface).radius(18.0f)
        .border(1.0f,kBorderSoft).hoverColor(kSurfaceRaised).onClick([tool]{state.tool=tool;state.status="就绪";}).build();
    ui.rect(id + ".icon.bg").position(x+22.0f,y+22.0f).size(44.0f,44.0f)
        .color(kSurfaceRaised).radius(13.0f).border(1.0f,kBorder).build();
    text(ui,id+".icon",mark,x+28.0f,y+32.0f,36.0f,22.0f,12.0f,accent,780);
    text(ui,id+".eyebrow",eyebrow,x+82.0f,y+23.0f,width-104.0f,18.0f,kFontOverline,accent,750);
    text(ui,id+".title",title,x+82.0f,y+43.0f,width-104.0f,28.0f,18.0f,kText,760);
    text(ui,id+".desc",description,x+22.0f,y+86.0f,width-44.0f,height-112.0f,kFontSecondary,kMuted,520,true);
    text(ui,id+".arrow","→",x+width-44.0f,y+height-42.0f,24.0f,24.0f,18.0f,kTextSecondary,650);
}

inline void composeHome(eui::Ui& ui, float x, float y, float width, float height) {
    text(ui,"home.hero.kicker","LOCAL AI, ONE WORKSPACE",x,y,width,18.0f,kFontOverline,kAccent,780);
    text(ui,"home.hero.title","在本机运行完整的生成与感知能力",x,y+26.0f,width,42.0f,28.0f,kText,800);
    text(ui,"home.hero.desc","从对话和语音到图像生成、目标检测、分割、姿态与深度估计。所有推理都由 GGML-Forge 在本地完成。",
         x,y+76.0f,std::min(width,820.0f),46.0f,kFontBody,kMuted,520,true);

    const float gap=16.0f;
    const float card_w=(width-gap)/2.0f;
    const float card_h=std::clamp((height-158.0f-gap)/2.0f,180.0f,260.0f);
    const float top=y+142.0f;
    homeCard(ui,"home.chat",x,top,card_w,card_h,"LANGUAGE","本地对话",
             "流式聊天、思考过程与 Agent 工具调用。","AI",Tool::Chat,kAccent);
    homeCard(ui,"home.speech",x+card_w+gap,top,card_w,card_h,"AUDIO","语音合成",
             "使用 GPT-SoVITS 生成并播放自然语音。","WAV",Tool::Speech,kPurple);
    homeCard(ui,"home.image",x,top+card_h+gap,card_w,card_h,"GENERATE","图像生成",
             "使用 Stable Diffusion 从文本生成图像。","IMG",Tool::Image,kOrange);
    homeCard(ui,"home.vision",x+card_w+gap,top+card_h+gap,card_w,card_h,"PERCEPTION","视觉分析",
             "分类、检测、分割、姿态、OBB 和深度估计。","CV",Tool::Vision,kBlue);
}

} // namespace app
