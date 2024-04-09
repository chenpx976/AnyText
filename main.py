import base64
import uvicorn
from fastapi import FastAPI, Body
from typing import Dict, Optional

import cv2
import gradio as gr
import numpy as np
from util import save_images
import argparse
import numpy as np
from typing import Dict, Any, Tuple


import cv2
import numpy as np
import requests
from io import BytesIO


from typing import Optional
from pydantic import BaseModel, HttpUrl, Field


from modelscope.pipelines import pipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_fp32",
        action="store_true",
        default=False,
        help="Whether or not to use fp32 during inference.",
    )
    parser.add_argument(
        "--no_translator",
        action="store_true",
        default=False,
        help="Whether or not to use the CH->EN translator, which enable input Chinese prompt and cause ~4GB VRAM.",
    )
    parser.add_argument(
        "--font_path",
        type=str,
        default="font/Arial_Unicode.ttf",
        help="path of a font file",
    )
    args = parser.parse_args()
    return args


inference = None

app = FastAPI()


@app.on_event("startup")
async def load_model():
    global inference
    args = parse_args()
    inference = pipeline(
        "my-anytext-task",
        model="damo/cv_anytext_text_generation_editing",
        model_revision="v1.1.2",
        use_fp16=not args.use_fp32,
        use_translator=not args.no_translator,
        font_path=args.font_path,
    )


def download_image(url):
    response = requests.get(url)
    image_bytes = BytesIO(response.content)
    image = cv2.imdecode(np.frombuffer(image_bytes.read(), np.uint8), cv2.IMREAD_COLOR)
    return image


def generate_pos_imgs(image):
    # 将图像转换为灰度图
    if len(image.shape) == 3 and image.shape[2] == 3:  # 检查是否为彩色图像
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    # 执行颜色反转
    inverted_img = 255 - gray_image
    # 将二维灰度图转换为三维彩色图像格式
    pos_imgs = cv2.cvtColor(inverted_img, cv2.COLOR_GRAY2BGR)
    return pos_imgs


def get_imgpos_from_url(url):
    print(f"Downloading image from {url}")
    image = download_image(url)
    pos_imgs = generate_pos_imgs(image)
    return pos_imgs


def convert_api_request_to_model_params(
    request_data: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    转换 API 请求数据为模型调用的入参格式。

    :param request_data: API 请求的数据。
    :return: 返回两个字典，第一个是模型的输入数据，第二个是模型的参数。
    """
    # 提取基本信息
    # mode = "gen" if "base_image_url" not in request_data["input"] else "edit"
    mode = "gen"
    prompt = request_data["input"]["prompt"]

    # 图片处理逻辑（例如下载、预处理）待实现
    mask_image_url = request_data["input"].get("mask_image_url")
    base_image_url = request_data["input"].get("base_image_url")
    if mask_image_url:
        pos_imgs = get_imgpos_from_url(mask_image_url)
    else:
        pos_imgs = None
    if base_image_url:
        ori_img = download_image(base_image_url)
    else:
        ori_img = None
    # 参数提取
    parameters = request_data.get("parameters", {})
    # layout_priority = parameters.get("layout_priority", "vertical")
    revise_pos = parameters.get("text_position_revise", False)
    n = parameters.get("n", 1)
    steps = parameters.get("steps", 20)
    w = parameters.get("image_width", 512)
    h = parameters.get("image_height", 512)
    strength = parameters.get("strength", 1.0)
    cfg_scale = parameters.get("cfg_scale", 9.0)
    eta = parameters.get("eta", 0.0)
    seed = parameters.get("seed", -1)
    a_prompt = request_data["input"].get("appended_prompt", "")
    n_prompt = request_data["input"].get("negative_prompt", "")

    # 构建模型输入数据
    input_data = {
        "prompt": prompt,
        "mode": mode,
        "seed": seed,
        "pos_imgs": pos_imgs,
        "ori_img": ori_img,
    }

    # 构建模型参数
    model_params = {
        "a_prompt": a_prompt
        or "best quality, extremely detailed,4k, HD, supper legible text,  clear text edges,  clear strokes, neat writing, no watermarks",
        "n_prompt": n_prompt
        or "low-res, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality, watermark, unreadable text, messy words, distorted text, disorganized writing, advertising picture",
        # "layout_priority": layout_priority,
        "revise_pos": revise_pos,
        "n": n,
        "steps": steps,
        "w": w,
        "h": h,
        "strength": strength,
        "cfg_scale": cfg_scale,
        "eta": eta,
    }

    return input_data, model_params


class InputModel(BaseModel):
    prompt: str = Field(..., example='一只浣熊站在黑板前，上面写着"深度学习"')
    mask_image_url: HttpUrl = Field(..., example="http://aliyun.com/a.jpg")
    base_image_url: Optional[HttpUrl] = Field(None, example="http://aliyun.com/a.jpg")
    appended_prompt: Optional[str] = Field(
        None,
        example="best quality, extremely detailed,4k, HD, supper legible text, clear text edges, clear strokes, neat writing, no watermarks",
    )
    negative_prompt: Optional[str] = Field(
        None,
        example="low-res, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality, watermark, unreadable text, messy words, distorted text, disorganized writing, advertising picture",
    )


class ParametersModel(BaseModel):
    layout_priority: Optional[str] = Field("vertical", example="vertical")
    text_position_revise: Optional[bool] = Field(False, example=False)
    n: Optional[int] = Field(1, example=1)
    steps: Optional[int] = Field(20, example=20)
    image_width: Optional[int] = Field(512, example=512)
    image_height: Optional[int] = Field(512, example=512)
    strength: Optional[float] = Field(1.0, example=1.0)
    cfg_scale: Optional[float] = Field(9.0, example=9.0)
    eta: Optional[float] = Field(0.0, example=0.0)
    seed: Optional[int] = Field(-1, example=-1)


class RequestModel(BaseModel):
    model: str = Field(..., example="wanx-anytext-v1")
    input: InputModel
    parameters: Optional[ParametersModel] = None


class OutputModel(BaseModel):
    result: Optional[Dict[str, Any]] = None


def model_process(
    prompt: str,
    seed: Optional[int] = -1,
    pos_imgs: Optional[np.ndarray] = None,
    ori_img: Optional[np.ndarray] = None,
    sort_radio: Optional[str] = "↔",
    show_debug: Optional[bool] = False,
    revise_pos: Optional[bool] = False,
    img_count: Optional[int] = 1,
    ddim_steps: Optional[int] = 20,
    w: Optional[int] = 512,
    h: Optional[int] = 512,
    strength: Optional[float] = 1.0,
    cfg_scale: Optional[float] = 9.0,
    eta: Optional[float] = 0.0,
    a_prompt: Optional[
        str
    ] = "best quality, extremely detailed,4k, HD, supper legible text, clear text edges, clear strokes, neat writing, no watermarks",
    n_prompt: Optional[
        str
    ] = "low-res, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality, watermark, unreadable text, messy words, distorted text, disorganized writing, advertising picture",
    mode: Optional[str] = "gen",
    img_save_folder: Optional[str] = "output",
    # 任何其他参数
    **kwargs,
):

    input_data = {
        "prompt": prompt,
        "seed": seed or -1,
        "draw_pos": pos_imgs,
        "ori_image": ori_img,
    }
    params = {
        "sort_priority": sort_radio or "↔",
        "show_debug": show_debug or False,
        "revise_pos": revise_pos,
        "image_count": img_count or 1,
        "ddim_steps": ddim_steps or 20,
        "image_width": w,
        "image_height": h,
        "strength": strength or 1,
        "cfg_scale": cfg_scale or 9,
        "eta": eta or 0,
        "a_prompt": a_prompt
        or "best quality, extremely detailed,4k, HD, supper legible text,  clear text edges,  clear strokes, neat writing, no watermarks",
        "n_prompt": n_prompt
        or "low-res, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality, watermark, unreadable text, messy words, distorted text, disorganized writing, advertising picture",
    }
    print(f"input_data: {input_data}")
    print(f"params: {params}")
    results, rtn_code, rtn_warning, debug_info = inference(
        input_data, mode=mode, **params
    )
    if rtn_code >= 0:
        save_images(results, img_save_folder)
        print(f"Done, result images are saved in: {img_save_folder}")
        if rtn_warning:
            gr.Warning(rtn_warning)
    else:
        raise gr.Error(rtn_warning)
    res = []
    if results is not None:
        for idx, img in enumerate(results):
            base64_img = base64.b64encode(
                cv2.imencode(".jpg", img[..., ::-1])[1]
            ).decode()
            res.append({"image": base64_img})
    return res


@app.post("/api/v1/services/aigc/anytext/generation", response_model=OutputModel)
async def generate_text(request: RequestModel = Body(...)):
    api_request = request.dict()
    input_data, model_params = convert_api_request_to_model_params(api_request)
    print("模型输入数据:", input_data)
    print("模型参数:", model_params)
    result = model_process(**input_data, **model_params)
    return OutputModel(
        result=result,
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=False)
