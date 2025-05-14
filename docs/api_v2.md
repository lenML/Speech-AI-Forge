### `v2/tts` API 文档

#### **接口描述**
`v2/tts` 是一个文本转语音（TTS）接口，支持多种输入格式（文本、SSML、多段文本），可基于参考音频进行语音克隆，并提供音频增强、调整、编码等功能。返回结果为音频文件流。

---

### **请求方式**
- **URL:** `/v2/tts`
- **方法:** `POST`
- **Content-Type:** `application/json`

---

### **请求参数**
请求体为 JSON 对象，支持以下字段：

#### **1. 输入文本（任选其一）**
| 参数名 | 类型 | 说明 |
|--------|------|------|
| `text` | `string` | 单段文本 |
| `texts` | `string[]` | 多段文本 |
| `ssml` | `string` | SSML 格式文本 |

#### **2. 说话人（spk）**
| 参数名 | 类型 | 说明 |
|--------|------|------|
| `spk.from_spk_id` | `string` | 预设说话人 ID |
| `spk.from_spk_name` | `string` | 预设说话人名称（如 "mona"） |
| `spk.from_ref.wav_b64` | `string` | 参考音频（Base64 编码） |
| `spk.from_ref.text` | `string` | 参考音频对应文本 |

#### **3. 语音调整（adjust）**
| 参数名 | 类型 | 说明 |
|--------|------|------|
| `adjust.pitch` | `number` | 音调调整（默认 `0`） |
| `adjust.speed_rate` | `number` | 语速调整（默认 `1`） |
| `adjust.volume_gain_db` | `number` | 音量调整（默认 `0`） |
| `adjust.normalize` | `boolean` | 是否归一化音量（默认 `false`） |
| `adjust.headroom` | `number` | 归一化动态余量（默认 `1`） |
| `adjust.remove_silence` | `boolean` | 是否移除音频两端的静音部分（默认 `false`） |
| `adjust.remove_silence_threshold` | `number` | 静音阈值（默认 `-42`） |

#### **4. 编码设置（encoder）**
| 参数名 | 类型 | 说明 |
|--------|------|------|
| `encoder.bitrate` | `string` | 比特率（如 `"64k"`） |
| `encoder.format` | `string` | 输出格式（如 `"mp3"`） |
| `encoder.acodec` | `string` | 音频编码（如 `"libmp3lame"`） |

#### **5. 音频增强（enhance）**
| 参数名 | 类型 | 说明 |
|--------|------|------|
| `enhance.enabled` | `boolean` | 是否启用增强（默认 `false`） |
| `enhance.model` | `string` | 增强模型（如 `"resemble-enhance"`） |
| `enhance.nfe` | `number` | 噪声滤波级别 |
| `enhance.solver` | `string` | 算法求解器（如 `"midpoint"`） |
| `enhance.lambd` | `number` | 增强参数（默认 `0.5`） |
| `enhance.tau` | `number` | 增强参数（默认 `0.5`） |

#### **6. 推理参数（infer）**
| 参数名 | 类型 | 说明 |
|--------|------|------|
| `infer.batch_size` | `number` | 处理批量大小 |
| `infer.spliter_threshold` | `number` | 句子拆分阈值 |
| `infer.eos` | `string` | 句子结束符（如 `"。"`） |
| `infer.seed` | `number` | 随机种子 |
| `infer.stream` | `boolean` | 是否启用流式输出 |
| `infer.stream_chunk_size` | `number` | 流式输出分块大小 |
| `infer.no_cache` | `boolean` | 是否禁用缓存 |
| `infer.sync_gen` | `boolean` | 是否同步生成 |

#### **7. 语音克隆（vc）**
> （即将废弃只是调用 openvoice 模型可能使用，但是不建议）

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `vc.enabled` | `boolean` | 是否启用语音克隆 |
| `vc.mid` | `string` | 语音克隆模型 |
| `vc.emotion` | `string` | 语音情感 |
| `vc.tau` | `number` | 语音克隆参数 |

#### **8. 文字正则化（TN）**
| 参数名 | 类型 | 说明 |
|--------|------|------|
| `tn.enabled` | `string[]` | 启用的正则化规则 |
| `tn.disabled` | `string[]` | 禁用的正则化规则 |

#### **9. TTS 相关参数（tts）**
| 参数名 | 类型 | 说明 |
|--------|------|------|
| `tts.mid` | `string` | TTS 模型（如 `"cosy-voice"`） |
| `tts.style` | `string` | 说话风格 |
| `tts.temperature` | `number` | 采样温度 |
| `tts.top_p` | `number` | 采样概率截断 |
| `tts.top_k` | `number` | 采样候选数量 |
| `tts.emotion` | `string` | 语音情感 |
| `tts.prompt` | `string` | 预设提示（即将废弃） |
| `tts.prompt1` | `string` | 备用预设提示（即将废弃） |
| `tts.prompt2` | `string` | 备用预设提示（即将废弃） |
| `tts.prefix` | `string` | 备用前缀（即将废弃） |

---

### **返回结果**
- **成功:** 音频文件流
- **失败:** `JSON` 错误信息

示例：
```json
{
  "error": "Invalid input text"
}
```

---

### **示例**

```bash
curl http://localhost:7870/v2/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, welcome to the Speech AI Forge project.",
    "spk": {
      "from_spk_name": "mona"
    },
    "adjust": {
      "pitch": 0,
      "speed_rate": 1
    },
    "encoder": {
      "bitrate": "64k",
      "format": "mp3",
      "acodec": "libmp3lame"
    },
    "enhance": {
      "enabled": true,
      "model": "resemble-enhance"
    },
    "tts": {
      "mid": "cosy-voice",
      "temperature": 0.3,
      "top_p": 0.75,
      "top_k": 20
    }
  }' \
  --output speech.mp3
```

#### 代码示例
可查看 ../examples/javascript/demo_tts_v2_full.mjs 文件

---

### **备注**
1. **文本输入格式**：`text`、`texts`、`ssml` 三者选其一，不可同时使用。
2. **语音克隆**：建议提供高质量参考音频，时长 3-5 秒。
3. **增强功能**：可能导致生成延迟，建议仅在必要时开启。
