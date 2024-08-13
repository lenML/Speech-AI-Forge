# Changelog

<a name="0.8.0-rc"></a>

## 0.8.0-rc (2024-08-13)

### Added

- ✨ add openvoice api [#100](https://github.com/lenML/ChatTTS-Forge/issues/100) [[0a7735a](https://github.com/lenML/ChatTTS-Forge/commit/0a7735afa301c9be6caae6f8a2e7442b02f610f0)]
- ✨ add openvoice model/dcls/downloader [#100](https://github.com/lenML/ChatTTS-Forge/issues/100) [[a14600d](https://github.com/lenML/ChatTTS-Forge/commit/a14600d00e4b2a6ec1801e196e3cf0799dcc2de0)]
- ✨ add webui ssml &#x60;&#x60;from_subtitle&#x60; tab [[3ab5d3f](https://github.com/lenML/ChatTTS-Forge/commit/3ab5d3fe23d36ac1313b8891ac4b19ba971b1b97)]
- ✨ add chattts txt_smp eos [#133](https://github.com/lenML/ChatTTS-Forge/issues/133) [[854e477](https://github.com/lenML/ChatTTS-Forge/commit/854e477b91d8661c6c359a6cb0f2a29cc7941a85)]
- ✨ add asr_tab output download [[26d9b0e](https://github.com/lenML/ChatTTS-Forge/commit/26d9b0edcf15b074d36c31a0f226769b60a26676)]
- ✨ add webui &#x60;asr&#x60; tab [[a2b0670](https://github.com/lenML/ChatTTS-Forge/commit/a2b0670c32ae56b4e7d24536db8bc0347eb4341c)]
- ✨ add webui &#x60;post process&#x60; tab [[6a7d9de](https://github.com/lenML/ChatTTS-Forge/commit/6a7d9de982074a22b6d0827a3f7aa8a590e93475)]
- ✨ improve &#x60;model_zoo&#x60; [[f04c1a4](https://github.com/lenML/ChatTTS-Forge/commit/f04c1a49e9efcd8e6dfa4f545714ba0a22a300b0)]
- ✨ add &#x60;/v1/stt/transcribe&#x60; api [[ccf21d3](https://github.com/lenML/ChatTTS-Forge/commit/ccf21d306f69446817d953698ee80bf19db71d08)]
- ✨ replace &#x60;openai.whisper&#x60; &#x3D;&gt; &#x60;faster_whisper&#x60; [[60d4d2c](https://github.com/lenML/ChatTTS-Forge/commit/60d4d2c1a331912d1d26316614e6d9c941f9d4a3)]
- ✨ add faster whisper downloader [[108d691](https://github.com/lenML/ChatTTS-Forge/commit/108d691e0989dda93dd3ee946cdaa7a733b20adc)]
- ✨ add script &#x60;get_spk_emb.py&#x60; [#128](https://github.com/lenML/ChatTTS-Forge/issues/128) [[e6a58c5](https://github.com/lenML/ChatTTS-Forge/commit/e6a58c51d172ae18ddc83a53bb79943247b0a36b)]
- ✨ add stt api [#92](https://github.com/lenML/ChatTTS-Forge/issues/92) [[92b992f](https://github.com/lenML/ChatTTS-Forge/commit/92b992f2a5f77c95696590dd1f6c8f548c6a687d)]
- ✨ impl whisper model [[ae3f416](https://github.com/lenML/ChatTTS-Forge/commit/ae3f4162d8e85173eda94112e6fdcf4513ad2a71)]
- ✨ api support specify model_id [[a486098](https://github.com/lenML/ChatTTS-Forge/commit/a486098ab0640b234cb639bbb920e1c176c36d65)]
- ✨ ChatTTS 支持 reference 推理 [#113](https://github.com/lenML/ChatTTS-Forge/issues/113) [[ff4991e](https://github.com/lenML/ChatTTS-Forge/commit/ff4991e0131ab38a929815dedbfbaf1e5b8f854a)]
- ✨ webui add Speaker Builder [[c5b5a27](https://github.com/lenML/ChatTTS-Forge/commit/c5b5a278ce787d783ed2b107e5e76c082cefc65f)]
- ✨ update chatts 下载脚本 [[702d63d](https://github.com/lenML/ChatTTS-Forge/commit/702d63d6828190e1e77e80c47df5005fb70dfcb0)]
- ✨ api support latest speaker structure [[e210f03](https://github.com/lenML/ChatTTS-Forge/commit/e210f03bed662117d4dc2a805d95436080386748)]
- ✨ add &#x60;/text/normalize&#x60; api [[ca0f7b2](https://github.com/lenML/ChatTTS-Forge/commit/ca0f7b25587a917bd314eb2885f5f1751c60588e)]
- ✨ add excepthook [[241bceb](https://github.com/lenML/ChatTTS-Forge/commit/241bcebbe642433ecd92b61358aaa4457f30510e)]
- ✨ support more tts models [#90](https://github.com/lenML/ChatTTS-Forge/issues/90) [[e3d1b14](https://github.com/lenML/ChatTTS-Forge/commit/e3d1b142ea5f80daf042cd10fa8a9b54dcd2893c)]
- ✨ add &#x60;CosyVoiceModel&#x60; [#90](https://github.com/lenML/ChatTTS-Forge/issues/90) [[3adb574](https://github.com/lenML/ChatTTS-Forge/commit/3adb5743802a16fc2fe96f2027f546ec59ce7703)]
- ✨ webui &#x60;tts_generate_with_history&#x60; [[58fe589](https://github.com/lenML/ChatTTS-Forge/commit/58fe589e85e8169dee47dcc5f2fb21f8a40f77ea)]
- ✨ webui min_temp &#x60;0.1&#x60; &#x3D;&gt; &#x60;0.000000001&#x60; [[3a41ef9](https://github.com/lenML/ChatTTS-Forge/commit/3a41ef9d3110d618f783c2c4d48646bc5b3400f0)]
- ✨ add xtts_v2 api &#x60;no_cache&#x60; [[a8d913d](https://github.com/lenML/ChatTTS-Forge/commit/a8d913d476d07c28135c5907fd6da36f0bec3036)]
- ✨ add cosyvoice spk file [[2854902](https://github.com/lenML/ChatTTS-Forge/commit/2854902fe0551dfed0292a5a24fc7e42b3779c92)]
- ✨ add spk edit script [[837b49b](https://github.com/lenML/ChatTTS-Forge/commit/837b49bc4fbc62d2085ed714ea9e9380bb33d8f3)]
- ✨ add cosyvoice config [[6556306](https://github.com/lenML/ChatTTS-Forge/commit/65563060de105e37b222521a2c44d237713423f7)]
- ✨ add cosyvoice downloader [[cc60120](https://github.com/lenML/ChatTTS-Forge/commit/cc60120076c3b880cbe3ced31227ce75cd979cae)]
- ✨ support fish speech model [[1835565](https://github.com/lenML/ChatTTS-Forge/commit/1835565997e2b8ed7f23c7e96bb3cda9ea41d59e)]
- ✨ refacotor style prompt [[9c8c0de](https://github.com/lenML/ChatTTS-Forge/commit/9c8c0de66106c283a669132f7a47fc738521858d)]
- ✨ add fishspeech model dl [[bdb3ee5](https://github.com/lenML/ChatTTS-Forge/commit/bdb3ee56086c819a377d1532e2739eb5735c2122)]
- ✨ improve tts handler [[8b85878](https://github.com/lenML/ChatTTS-Forge/commit/8b85878a48cb2971cabbd69044b5a6ab965c168c)]
- ✨ tn 兼容 windows [[0919b01](https://github.com/lenML/ChatTTS-Forge/commit/0919b0143bf2c872e4335493fa3b7492fd0963e4)]
- ✨ webui in &#x60;tts_pipeline&#x60; [[ea39d6b](https://github.com/lenML/ChatTTS-Forge/commit/ea39d6b64784f393a430171df5c33ac7eb0a5c94)]
- ✨ streaming encoder [[50d41a9](https://github.com/lenML/ChatTTS-Forge/commit/50d41a9491913d24705b470bcf10d5a438906492)]
- ✨ 完善 stream mode [[3095d4c](https://github.com/lenML/ChatTTS-Forge/commit/3095d4cd8067fbc6dda95bd6a6278c5b324f4cf1)]
- ✨ add InferCache [[fd93e79](https://github.com/lenML/ChatTTS-Forge/commit/fd93e79537d71d3fb6a7e925f61eaacddee80c18)]
- ✅ pass TN test cases [[f604f5c](https://github.com/lenML/ChatTTS-Forge/commit/f604f5cdb6080adb64e04109016df8bd03e4ad95)]
- ✨ models load by zoo [[7835eb6](https://github.com/lenML/ChatTTS-Forge/commit/7835eb61c34fc2371be75f11604bb1030a8cdf47)]
- ✨ fit tests case to &#x60;dev_tts_pipeline&#x60; [[b24cb17](https://github.com/lenML/ChatTTS-Forge/commit/b24cb1761ee02df22df5181349e54dc4c326392e)]
- ✨ fit api &#x3D;&gt; tts pipeline [[c4baeb1](https://github.com/lenML/ChatTTS-Forge/commit/c4baeb166337baf371800b2235ed14647ac96d21)]
- ✨ add &#x60;--train_mse&#x60; [[7e54bb7](https://github.com/lenML/ChatTTS-Forge/commit/7e54bb7482465010ac960440ac7a343ca35e202b)]

### Changed

- 🍱 add spk file [[cd338c2](https://github.com/lenML/ChatTTS-Forge/commit/cd338c23c871b18e3740a11dce4588dcc48b1f38)]
- 📌 pin openvoice [#100](https://github.com/lenML/ChatTTS-Forge/issues/100) [[b7bbd91](https://github.com/lenML/ChatTTS-Forge/commit/b7bbd91b7e32388d28f2d3aaa1bfdc560dca31bd)]
- 🎨 run format [[82fbc13](https://github.com/lenML/ChatTTS-Forge/commit/82fbc13441a7067faeafa3d079bc3218c1eed5a5)]
- 🎨 run format [[318d33f](https://github.com/lenML/ChatTTS-Forge/commit/318d33f8d0b1451a39b3cbc94debca7f4f21dfca)]
- 📌 sync upstream [#113](https://github.com/lenML/ChatTTS-Forge/issues/113) [[5d4bafc](https://github.com/lenML/ChatTTS-Forge/commit/5d4bafc6e17c1f124f606159655b127a0870b942)]
- 📌 pin repos_static.ChatTTS [[b1102ed](https://github.com/lenML/ChatTTS-Forge/commit/b1102eda29178bc89635180ee3b43b8af498e162)]
- 📌 pin markedjs [[20f6781](https://github.com/lenML/ChatTTS-Forge/commit/20f6781d1373edfe063e57bcb6639af0c0dec22e)]
- 📌 pin cosyvoice [[54f59e8](https://github.com/lenML/ChatTTS-Forge/commit/54f59e8b4f2c8a3e4a2afcb9ea8dcbaf7fee4b5b)]
- 📌 pin fishspeech rename [[af9efdf](https://github.com/lenML/ChatTTS-Forge/commit/af9efdfd2ca00216afc0db9ef9d378a41a8d44ee)]
- 📌 pin fishspeech [[c8905df](https://github.com/lenML/ChatTTS-Forge/commit/c8905df4fc7dc780104144728773b1c4ce655f95)]
- 🎨 run format [[246fd25](https://github.com/lenML/ChatTTS-Forge/commit/246fd25d7ab920ab7a2e7980b3d7c4dd83ffe8b7)]
- ♻️ refactor TTSSpeaker [#90](https://github.com/lenML/ChatTTS-Forge/issues/90) [[7d7f465](https://github.com/lenML/ChatTTS-Forge/commit/7d7f4652e88d97384c289dc30f9fc2a8d400a351)]
- 🎨 run format [[21c2760](https://github.com/lenML/ChatTTS-Forge/commit/21c2760c1024562cb1a5a866ec5516fffab7e263)]
- ⬆️ Bump certifi from 2024.2.2 to 2024.7.4 [[006f246](https://github.com/lenML/ChatTTS-Forge/commit/006f24685e1089c09909556797334bae25e1fd07)]

### Breaking changes

- 💥 breaking improve [#91](https://github.com/lenML/ChatTTS-Forge/issues/91) [[3d62765](https://github.com/lenML/ChatTTS-Forge/commit/3d62765eb8173527be993fcee58fab668bfae1fd)]

### Removed

- 🔥 remove test files [[b1d5d69](https://github.com/lenML/ChatTTS-Forge/commit/b1d5d6931c5bf2c9496b3c9483ccc8b931d9e035)]
- 🔥 remove old speaker files [[caf8787](https://github.com/lenML/ChatTTS-Forge/commit/caf87872652d5a9adc33fac0ea054b43e44c15e3)]

### Fixed

- 🐛 fix chattts audio resample/cache [[02bf989](https://github.com/lenML/ChatTTS-Forge/commit/02bf98966b07fbd790fffeb9a57aafe101e5bfb3)]
- 🐛 fix webui progress missing [[8d118ee](https://github.com/lenML/ChatTTS-Forge/commit/8d118ee46204e8e9c3569507f1a3dfa2bfab53e3)]
- 🐛 fix whisper resample [[55b79f2](https://github.com/lenML/ChatTTS-Forge/commit/55b79f2103534ea5b9b9f72c49181a2ce1f50fdb)]
- 🐛 fix typo [[74941b4](https://github.com/lenML/ChatTTS-Forge/commit/74941b4540dccf76babdb801d6cfd563e8a9019e)]
- 🐛 fix tests &#x60;openai_api_stt&#x60; [[d0a8e09](https://github.com/lenML/ChatTTS-Forge/commit/d0a8e09ca67e01c254698dfb54cb9cbbfa344936)]
- 🐛 fix wrong inport [[8d8434f](https://github.com/lenML/ChatTTS-Forge/commit/8d8434f4b0e423f760833f31aa29c804871f28be)]
- 🐛 fix get_spk_emb script [[f392260](https://github.com/lenML/ChatTTS-Forge/commit/f3922603b9e13ec61f1773f6ccd779399e077500)]
- 🐛 pin gradio version / fix &#x60;.pt&#x60; &#x3D;&gt; &#x60;.json [#126](https://github.com/lenML/ChatTTS-Forge/issues/126) [[bbc19fb](https://github.com/lenML/ChatTTS-Forge/commit/bbc19fba7ba55380c4241383cd15891fc7a911a0)]
- 🐛 fix model loader [[d904679](https://github.com/lenML/ChatTTS-Forge/commit/d904679f7361508904ff14d701c59900055988f5)]
- 🐛 fix mps error msg [[928a9b4](https://github.com/lenML/ChatTTS-Forge/commit/928a9b44ae74b82dfec986ddec67785a068159bf)]
- 🐛 fix generate function precision [#120](https://github.com/lenML/ChatTTS-Forge/issues/120) [[77e6eeb](https://github.com/lenML/ChatTTS-Forge/commit/77e6eebc57265a1ffeff7c28197e5511395e117e)]
- 🐛 fix tokenizer.padding_side [[e46379c](https://github.com/lenML/ChatTTS-Forge/commit/e46379cdcc3b071a356fb9ad07a129d6303ddb18)]
- 🐛 fix test case in &#x60;spk&#x3D;&#x3D;None&#x60; [[ff26e67](https://github.com/lenML/ChatTTS-Forge/commit/ff26e67a4aaf9ca636007b460d17059940640af4)]
- 🐛 fix refine [#117](https://github.com/lenML/ChatTTS-Forge/issues/117) [[f0c2602](https://github.com/lenML/ChatTTS-Forge/commit/f0c26024b150859c75d39b1e93b90068fcf2362c)]
- 🐛 fix execpthook [[a9c9d16](https://github.com/lenML/ChatTTS-Forge/commit/a9c9d16c86201e68089b275062d5a076e27f62e8)]
- 🐛 fix playground pages [[d106390](https://github.com/lenML/ChatTTS-Forge/commit/d106390f51b17aca26418dd61f13ef8f10f2a8ab)]
- 🐛 fix refine api [#116](https://github.com/lenML/ChatTTS-Forge/issues/116) [[86ae55a](https://github.com/lenML/ChatTTS-Forge/commit/86ae55a27dafa5e55f4243b525dabfabe0dc9550)]
- 🐛 improve generator gc [[3b45772](https://github.com/lenML/ChatTTS-Forge/commit/3b4577293e52101de32ed824bf50dee2f3600d56)]
- 🐛 fix cosy voice model loader [[fb3df64](https://github.com/lenML/ChatTTS-Forge/commit/fb3df6408e4a070c7911565537ce83583e554831)]
- 🐛 fix spk file api [#114](https://github.com/lenML/ChatTTS-Forge/issues/114) [[c0a1373](https://github.com/lenML/ChatTTS-Forge/commit/c0a13734ee331d5541c9a027d13adbbaa4fc5210)]
- 🐛 add &#x60;--off_track_tqdm&#x60; fix [#110](https://github.com/lenML/ChatTTS-Forge/issues/110) [[e57b37c](https://github.com/lenML/ChatTTS-Forge/commit/e57b37cd1884afcbf0c939b275e8177200ebb4e1)]
- 🐛 fix downloader [[d0c1467](https://github.com/lenML/ChatTTS-Forge/commit/d0c1467d29abc9f1a9b1d35d56ab1587ad8eae82)]
- 🐛 fix &#x60;StreamEncoder&#x60; reader mode [#108](https://github.com/lenML/ChatTTS-Forge/issues/108) [[3613d0d](https://github.com/lenML/ChatTTS-Forge/commit/3613d0db54e692c6dfbfe7697be2411b7e8ef7f9)]
- 🐛 compatible with py3.11 [#106](https://github.com/lenML/ChatTTS-Forge/issues/106) [[8f4d8b6](https://github.com/lenML/ChatTTS-Forge/commit/8f4d8b6653b7870495f1be6b19fa9d771b89780c)]
- 🐛 fix webui bug [#105](https://github.com/lenML/ChatTTS-Forge/issues/105) [[5c6b098](https://github.com/lenML/ChatTTS-Forge/commit/5c6b0983d36ea71b18e8b7360220a2665a4aac3a)]
- 🐛 fix stream encoder header [#103](https://github.com/lenML/ChatTTS-Forge/issues/103) [[f7312c1](https://github.com/lenML/ChatTTS-Forge/commit/f7312c11cd8e3d156dbff62367fa725d95aeaa94)]
- 🐛 fix file encoding [[c5006fb](https://github.com/lenML/ChatTTS-Forge/commit/c5006fb5d76260443727220875dd69efd4abc866)]
- 🐛 tn 兼容 windows [[bbada30](https://github.com/lenML/ChatTTS-Forge/commit/bbada30c9e10e2119a45af09f99cc3af17c1b20f)]
- 🐛 修复 playground ssml page error [[112a90f](https://github.com/lenML/ChatTTS-Forge/commit/112a90f01ba69666077ec8395bfc30bf832e235e)]
- 🐛 修复 style/spk 计算逻辑 [[62ed997](https://github.com/lenML/ChatTTS-Forge/commit/62ed99750d419d511181a72fcb18b2a797b32d1b)]
- 🐛 fix &#x60;ensure_non_empty&#x60; issues [#99](https://github.com/lenML/ChatTTS-Forge/issues/99) [[bfc3789](https://github.com/lenML/ChatTTS-Forge/commit/bfc37897eca494129d462473cdcb8ec899f00ba0)]
- 🐛 manual pick ee07d82f2155c56ee31340f5c073a2bef64ba55c [[6d857a7](https://github.com/lenML/ChatTTS-Forge/commit/6d857a72b0e88a6afd7a331b58a980bba3bd8cb2)]
- 🐛 manual pick 906f2f78cc4a71b59dcf83f4c7fea33e7a637127 [[878fa52](https://github.com/lenML/ChatTTS-Forge/commit/878fa52aa969a9a9b924926cbfa53c907324d07a)]
- 🐛 fix api test cases [[0953816](https://github.com/lenML/ChatTTS-Forge/commit/09538162117942fe0d18598cbc39113d53db8de0)]
- 🐛 fix generate dtype [[ee07d82](https://github.com/lenML/ChatTTS-Forge/commit/ee07d82f2155c56ee31340f5c073a2bef64ba55c)]
- 🐛 fix ensure memory usage [#99](https://github.com/lenML/ChatTTS-Forge/issues/99) [[906f2f7](https://github.com/lenML/ChatTTS-Forge/commit/906f2f78cc4a71b59dcf83f4c7fea33e7a637127)]

### Miscellaneous

- Merge pull request [#136](https://github.com/lenML/ChatTTS-Forge/issues/136) from fangd123/main [[fbea889](https://github.com/lenML/ChatTTS-Forge/commit/fbea8893c22483784324d4ab99909a170ccc25b5)]
- update the instruction of installing WeTextProcessing [[c816d67](https://github.com/lenML/ChatTTS-Forge/commit/c816d67859a5bfed4206e24484efae78a1c18fa4)]
- replace chattts modelscope_repo with AI-ModelScope [[0f76bce](https://github.com/lenML/ChatTTS-Forge/commit/0f76bce0e2ccca9fc97e5cfd7c08c870d0e41463)]
- fix: ensure SpkEncoder tensor with dtype.float [[b9491a6](https://github.com/lenML/ChatTTS-Forge/commit/b9491a6140d6a8c9a41ccecb751e4d9560e859e4)]
- fix: topK and topP not valid [#115](https://github.com/lenML/ChatTTS-Forge/issues/115) [[3ce2424](https://github.com/lenML/ChatTTS-Forge/commit/3ce2424993fd585b46a5931df76bc91a370d8e2e)]
- Merge pull request [#115](https://github.com/lenML/ChatTTS-Forge/issues/115) from wenyangchou/main [[f9cb075](https://github.com/lenML/ChatTTS-Forge/commit/f9cb0751696fa1e241840318c99dbee5c82ca94f)]
- fix: topK and topP not valid [[c0bfe02](https://github.com/lenML/ChatTTS-Forge/commit/c0bfe022fb131e0cddf7a0ccea67e9b005025b24)]
- Merge pull request [#102](https://github.com/lenML/ChatTTS-Forge/issues/102) from lenML/dev_tts_pipeline [[4028da1](https://github.com/lenML/ChatTTS-Forge/commit/4028da172e970e7bfeb0e8d2eb41d433881f9bdc)]
- merge main into dev_tts_pipeline [[846d18f](https://github.com/lenML/ChatTTS-Forge/commit/846d18f4992d1ba39c525072c8974172192587e5)]
- Merge pull request [#93](https://github.com/lenML/ChatTTS-Forge/issues/93) from lenML/dependabot/pip/certifi-2024.7.4 [[2f252df](https://github.com/lenML/ChatTTS-Forge/commit/2f252df6cd616a1b633224a1b34b71984947e90d)]

<a name="0.7.0"></a>

## 0.7.0 (2024-07-04)

### Added

- ✨ add stream mode to openai api [[989c3c5](https://github.com/lenML/ChatTTS-Forge/commit/989c3c500e9dd0e671501386202672be03801071)]
- ✨ add &#x60;PYTORCH_ENABLE_MPS_FALLBACK&#x60; [[2024e0b](https://github.com/lenML/ChatTTS-Forge/commit/2024e0b5c3f3212265a6ad6669235d272fee353b)]
- ✨ add &#x60;--flash_attn&#x60; args [[a449091](https://github.com/lenML/ChatTTS-Forge/commit/a4490910c30c88d91cd81ecfb8d84889b58c8415)]
- ✅ add stream test cases [[02fbb71](https://github.com/lenML/ChatTTS-Forge/commit/02fbb71e5af1b78dd43e84fe68d73708ca5df38e)]
- ✨ improve refine ui [[4676c71](https://github.com/lenML/ChatTTS-Forge/commit/4676c717614ad36c44a8be3bf59de9f8e5fe9400)]
- ✨ improve infer [[f9b81aa](https://github.com/lenML/ChatTTS-Forge/commit/f9b81aa4ae04ee7d4c1709c9c796d7e7a1f894d0)]
- ✨ improve generate [[b1f13b5](https://github.com/lenML/ChatTTS-Forge/commit/b1f13b5c95de0e6a806b1793259a9d2412569339)]
- ✨ improve podcast tab [[fa63491](https://github.com/lenML/ChatTTS-Forge/commit/fa63491a5590c73db852c65d819efebc0a33a409)]
- ✨ improve webui split_tab [[9578cc7](https://github.com/lenML/ChatTTS-Forge/commit/9578cc7a76a45fc9c470abe906fe5249b160968a)]
- ✨ SentenceSplitter use tokenizer [[d8b8596](https://github.com/lenML/ChatTTS-Forge/commit/d8b8596c2ae7fbeba6482c2ed21437a4b956ed5c)]
- ✨ add warning add docs [[7370ba2](https://github.com/lenML/ChatTTS-Forge/commit/7370ba28bd1561925b28e1bdd357deaf515eaa08)]
- ✨ add adjuster to webui [[01f09b4](https://github.com/lenML/ChatTTS-Forge/commit/01f09b4fad2eb8b24a16b7768403de4975d51774)]
- ✨ stream mode support adjuster [[585d2dd](https://github.com/lenML/ChatTTS-Forge/commit/585d2dd488d8f8387e0d9435fb399f090a41b9cc)]
- ✨ improve xtts_v2 api [[fec66c7](https://github.com/lenML/ChatTTS-Forge/commit/fec66c7c00939a3c7c15e007536e037ac01153fa)]
- ✨ improve normalize [[d0da37e](https://github.com/lenML/ChatTTS-Forge/commit/d0da37e43f1de4088ef638edd90723f93894b1d2)]
- ✨ improve normalize/spliter [[163b649](https://github.com/lenML/ChatTTS-Forge/commit/163b6490e4d453c37cc259ce27208f55d10a9084)]
- ✨ add loudness equalization [[bc8bda7](https://github.com/lenML/ChatTTS-Forge/commit/bc8bda74825c31985d3cc1a44366ad92af1b623a)]
- ✨ support &#x60;--use_cpu&#x3D;chattts,enhancer,trainer,all&#x60; [[23023bc](https://github.com/lenML/ChatTTS-Forge/commit/23023bc610f6f74a157faa8a6c6aacf64d91d870)]
- ✨ improve normalizetion.py [[1a7c0ed](https://github.com/lenML/ChatTTS-Forge/commit/1a7c0ed3923234ceadb79f397fa7577f9e682f2d)]
- ✨ ignore_useless_warnings [[4b9a32e](https://github.com/lenML/ChatTTS-Forge/commit/4b9a32ef821d85ceaf3d62af8f871aeb5088e084)]
- ✨ enhance logger, info &#x3D;&gt; debug [[73bc8e7](https://github.com/lenML/ChatTTS-Forge/commit/73bc8e72b40146debd0a59100b1cca4cc42f5029)]
- ✨ add playground.stream page [[31377b0](https://github.com/lenML/ChatTTS-Forge/commit/31377b060c182519d74a12d81e66c8e73686bcd8)]
- ✨ tts api support stream [#5](https://github.com/lenML/ChatTTS-Forge/issues/5) [[15e0b2c](https://github.com/lenML/ChatTTS-Forge/commit/15e0b2cb051ba39dcf99f60f1faa11941f6dc656)]

### Changed

- ⬆️ sync upstream changes [[5d174b3](https://github.com/lenML/ChatTTS-Forge/commit/5d174b39b594861f44507c6b50d615d8d3886796)]
- ⬆️ sync upstream changes [[e49dbb8](https://github.com/lenML/ChatTTS-Forge/commit/e49dbb8efad6122d6f7510df412edaa6188af83f)]
- 🎨 format [[f5f483a](https://github.com/lenML/ChatTTS-Forge/commit/f5f483a4c58d9be3da1f2bd8694d190d1a46e37e)]
- ⬆️ sync upstream changes for ALL [[f7fb126](https://github.com/lenML/ChatTTS-Forge/commit/f7fb1262fa1b91f70e4d4dcf7cc0c66f0c01fcb6)]
- ⬆️ sync upstream changes for &#x60;dvae.py&#x60; [[cc3ca09](https://github.com/lenML/ChatTTS-Forge/commit/cc3ca09eac12aea4dfa01cfa12ae43bcd3018329)]
- 🎨 format [[6af9e24](https://github.com/lenML/ChatTTS-Forge/commit/6af9e24de0e0bade35a33c6c3e68e29594ba0c3b)]
- ♻️ refactor SentenceSplitter [[d90c862](https://github.com/lenML/ChatTTS-Forge/commit/d90c862748f6657569baf775d5b168a595e046df)]
- ♻️ refactor models_setup [[ff9c7c0](https://github.com/lenML/ChatTTS-Forge/commit/ff9c7c08ba938e7ca04ecd5c67655334d9704d03)]
- 🍱 add \_p_en [[56f1fbf](https://github.com/lenML/ChatTTS-Forge/commit/56f1fbf1f3fff6f76ca8c29aa12a6ddef665cf9f)]
- 🍱 update prompt [[4f95b31](https://github.com/lenML/ChatTTS-Forge/commit/4f95b31679225e1ee144a411a9cfa9b30c598450)]
- ⚡ Reduce popping sounds [[2d0fd68](https://github.com/lenML/ChatTTS-Forge/commit/2d0fd688ad1a5cff1e6aafc0502aee26de3f1d75)]
- ⚡ improve &#x60;apply_character_map&#x60; [[ea7399f](https://github.com/lenML/ChatTTS-Forge/commit/ea7399facc5c29327a7870bd66ad6222f5731ce3)]

### Fixed

- 🐛 fix straem generate [[7f19d4f](https://github.com/lenML/ChatTTS-Forge/commit/7f19d4fb3aee8806030d4f6eaf910328ab16d629)]
- 🐛 add map_location [[a5f90cb](https://github.com/lenML/ChatTTS-Forge/commit/a5f90cba13a958e81d234a593ac60aca3d956029)]
- 🐛 fix dve indices.dtype [[db1e571](https://github.com/lenML/ChatTTS-Forge/commit/db1e57191e51c5a680227d732b42da33036ce1e9)]
- 🐛 fix missing &#x60;trange&#x60; [[8577a53](https://github.com/lenML/ChatTTS-Forge/commit/8577a5376d6c32b4ccfce4990018527b068a13ce)]
- 🐛 fix Including &#x60;&amp;&#x60; escape char causes normalization errors [#77](https://github.com/lenML/ChatTTS-Forge/issues/77) [[85c98f6](https://github.com/lenML/ChatTTS-Forge/commit/85c98f69c149863fb7d6bb962110cdd034830885)]
- 🐛 fix speaker loader [#71](https://github.com/lenML/ChatTTS-Forge/issues/71) [[e7b759f](https://github.com/lenML/ChatTTS-Forge/commit/e7b759f7765627a768b2feaf932e39b827137346)]
- 🐛 fix load speaker from seed [#69](https://github.com/lenML/ChatTTS-Forge/issues/69) [[304c318](https://github.com/lenML/ChatTTS-Forge/commit/304c31886bbbcd2a389e66184a21620baadd24dd)]
- 🐛 fix apply_prosody [[7fa55d9](https://github.com/lenML/ChatTTS-Forge/commit/7fa55d90c27399a71abfb89ab2a3b6514327dbde)]
- 🐛 fix normalization lang detect [[bd5e6eb](https://github.com/lenML/ChatTTS-Forge/commit/bd5e6eb88930b6c537978b4f9b89a92b2d8f21cb)]
- 🐛 remove rubberband-cli dependencies [#68](https://github.com/lenML/ChatTTS-Forge/issues/68) unit test [[650a668](https://github.com/lenML/ChatTTS-Forge/commit/650a668e89d8ebbebf637ce27a46f05f6b45f1ac)]
- 🐛 remove rubberband-cli dependencies [#68](https://github.com/lenML/ChatTTS-Forge/issues/68) [[1cd34c3](https://github.com/lenML/ChatTTS-Forge/commit/1cd34c32190d0787d22f73def9d8b69d6dfb4ea5)]
- 🐛 fix &#x60;apply_normalize&#x60; missing &#x60;sr&#x60; [[2db6d65](https://github.com/lenML/ChatTTS-Forge/commit/2db6d65ef8fbf8a3a213cbdc3d4b1143396cc165)]
- 🐛 fix sentence spliter [[5d8937c](https://github.com/lenML/ChatTTS-Forge/commit/5d8937c169d5f7784920a93834df0480dd3a67b3)]
- 🐛 fix playground url_join [[53e7cbc](https://github.com/lenML/ChatTTS-Forge/commit/53e7cbc6103bc0e3bb83767a9233c45285b77e75)]
- 🐛 fix generate_audio args [[a7a698c](https://github.com/lenML/ChatTTS-Forge/commit/a7a698c760b5bc97c90a144a4a7afb5e17414995)]
- 🐛 fix infer func [[b0de527](https://github.com/lenML/ChatTTS-Forge/commit/b0de5275342c02d332a50d0ab5ac171a7007b300)]
- 🐛 fix webui logging format [[4adc29e](https://github.com/lenML/ChatTTS-Forge/commit/4adc29e6c06fa806a8178f445399bbac8ed57911)]
- 🐛 fix webui speaker_tab missing progress [[fafe242](https://github.com/lenML/ChatTTS-Forge/commit/fafe242e69ea8019729a62e52f6c0b3c0d6a63ad)]

### Miscellaneous

- Merge pull request [#84](https://github.com/lenML/ChatTTS-Forge/issues/84) from wenyangchou/main [[f811e3d](https://github.com/lenML/ChatTTS-Forge/commit/f811e3dfe2d80876706fec2162dc664165e56a19)]
- Update Dockerfile [[e0cc31c](https://github.com/lenML/ChatTTS-Forge/commit/e0cc31cfd143c228f9d40ef1bb0f161666105e22)]
- optimize docker build [[7b187b6](https://github.com/lenML/ChatTTS-Forge/commit/7b187b6ac45e99d7e9bccd5dba1421565f27dcb2)]
- 🔨 add download_audio_backend.py [[8dd6925](https://github.com/lenML/ChatTTS-Forge/commit/8dd6925699bc1d01196f84bf76f2c0cee40f89ee)]
- 💩 revert libsora &#x3D;&gt; pyrubberband [[4ead989](https://github.com/lenML/ChatTTS-Forge/commit/4ead98966ed8c8f881ef021040612daa39a96585)]
- Merge pull request [#66](https://github.com/lenML/ChatTTS-Forge/issues/66) from WannaTen/main [[9af0361](https://github.com/lenML/ChatTTS-Forge/commit/9af0361e7d8860ddf97dc5a504cb4c3d8905a423)]
- fix port in api mode [[40ab68b](https://github.com/lenML/ChatTTS-Forge/commit/40ab68b624ea33ea51e7ba13b7a30a83bec52826)]
- Windows not yet supported for torch.compile fix [[74ac27d](https://github.com/lenML/ChatTTS-Forge/commit/74ac27d56a370f87560329043c42be27022ca0f5)]
- fix: replace mispronounced words in TTS [[de66e6b](https://github.com/lenML/ChatTTS-Forge/commit/de66e6b8f7f8b5c10e7ac54f7b2488c798e5ef81)]
- feat: support stream mode [[3da0f0c](https://github.com/lenML/ChatTTS-Forge/commit/3da0f0cb7f213dee40d00a89093166ad9e1d17a0)]
- optimize: mps audio quality by contiguous scores [[1e4d79f](https://github.com/lenML/ChatTTS-Forge/commit/1e4d79f1a81a3ac8697afff0e44f0cfd2608599a)]

<a name="0.6.1"></a>

## 0.6.1 (2024-06-18)

### Added

- ✨ add &#x60;--preload_models&#x60; [[73a41e0](https://github.com/lenML/ChatTTS-Forge/commit/73a41e009cd4426dfe4b0a35325da68189966390)]
- ✨ add webui progress [[778802d](https://github.com/lenML/ChatTTS-Forge/commit/778802ded12de340520f41a3e1bdb852f00bd637)]
- ✨ add merger error [[51060bc](https://github.com/lenML/ChatTTS-Forge/commit/51060bc343a6308493b7d582e21dca62eacaa7cb)]
- ✨ tts prompt &#x3D;&gt; experimental [[d3e6315](https://github.com/lenML/ChatTTS-Forge/commit/d3e6315a3cb8b1fa254cefb2efe2bae7c74a50f8)]
- ✨ add 基本的 speaker finetune ui [[5f68f19](https://github.com/lenML/ChatTTS-Forge/commit/5f68f193e78f470bd2c3ca4b9fa1008cf809e753)]
- ✨ add speaker finetune [[5ce27ed](https://github.com/lenML/ChatTTS-Forge/commit/5ce27ed7e4da6c96bb3fd016b8b491768faf319d)]
- ✨ add &#x60;--ino_half&#x60; remove &#x60;--half&#x60; [[5820e57](https://github.com/lenML/ChatTTS-Forge/commit/5820e576b288df50b929fbdfd9d0d6b6f548b54e)]
- ✨ add webui podcast 默认值 [[dd786a8](https://github.com/lenML/ChatTTS-Forge/commit/dd786a83733a71d005ff7efe6312e35d652b2525)]
- ✨ add webui 分割器配置 [[589327b](https://github.com/lenML/ChatTTS-Forge/commit/589327b729188d1385838816b9807e894eb128b0)]
- ✨ add &#x60;eos&#x60; params to all api [[79c994f](https://github.com/lenML/ChatTTS-Forge/commit/79c994fadf7d60ea432b62f4000b62b67efe7259)]

### Changed

- ⬆️ Bump urllib3 from 2.2.1 to 2.2.2 [[097c15b](https://github.com/lenML/ChatTTS-Forge/commit/097c15ba56f8197a4f26adcfb77336a70e5ed806)]
- 🎨 run formatter [[8c267e1](https://github.com/lenML/ChatTTS-Forge/commit/8c267e151152fe2090528104627ec031453d4ed5)]
- ⚡ Optimize &#x60;audio_data_to_segment&#x60; [#57](https://github.com/lenML/ChatTTS-Forge/issues/57) [[d33809c](https://github.com/lenML/ChatTTS-Forge/commit/d33809c60a3ac76a01f71de4fd26b315d066c8d3)]
- ⚡ map_location&#x3D;&quot;cpu&quot; [[0f58c10](https://github.com/lenML/ChatTTS-Forge/commit/0f58c10a445efaa9829f862acb4fb94bc07f07bf)]
- ⚡ colab use default GPU [[c7938ad](https://github.com/lenML/ChatTTS-Forge/commit/c7938adb6d3615f37210b1f3cbe4671f93d58285)]
- ⚡ improve hf calling [[2dde612](https://github.com/lenML/ChatTTS-Forge/commit/2dde6127906ce6e77a970b4cd96e68f7a5417c6a)]
- 🍱 add &#x60;bob_ft10.pt&#x60; [[9eee965](https://github.com/lenML/ChatTTS-Forge/commit/9eee965425a7d6640eba22d843db4975dd3e355a)]
- ⚡ enhance SynthesizeSegments [[0bb4dd7](https://github.com/lenML/ChatTTS-Forge/commit/0bb4dd7676c38249f10bf0326174ff8b74b2abae)]
- 🍱 add &#x60;bob_ft10.pt&#x60; [[bef1b02](https://github.com/lenML/ChatTTS-Forge/commit/bef1b02435c39830612b18738bb31ac48e340fc6)]
- ♻️ refactor api [[671fcc3](https://github.com/lenML/ChatTTS-Forge/commit/671fcc38a570d0cb7de0a214d318281084c9608c)]
- ⚡ improve xtts_v2 api [[206fabc](https://github.com/lenML/ChatTTS-Forge/commit/206fabc76f1dbad261c857cb02f8c99c21e99eef)]
- ⚡ train text &#x3D;&gt; just text [[e2037e0](https://github.com/lenML/ChatTTS-Forge/commit/e2037e0f97f15ff560fce14bbdc3926e3261bff9)]
- ⚡ improve TN [[a0069ed](https://github.com/lenML/ChatTTS-Forge/commit/a0069ed2d0c3122444e873fb13b9922f9ab88a79)]

### Fixed

- 🐛 fix webui speaker_editor missing &#x60;describe&#x60; [[2a2a36d](https://github.com/lenML/ChatTTS-Forge/commit/2a2a36d62d8f253fc2e17ccc558038dbcc99d1ee)]
- 💚 Dependabot alerts [[f501860](https://github.com/lenML/ChatTTS-Forge/commit/f5018607f602769d4dda7aa00573b9a06e659d91)]
- 🐛 fix &#x60;numpy&lt;2&#x60; [#50](https://github.com/lenML/ChatTTS-Forge/issues/50) [[e4fea4f](https://github.com/lenML/ChatTTS-Forge/commit/e4fea4f80b31d962f02cd1146ce8c73bf75b6a39)]
- 🐛 fix Box() index [#49](https://github.com/lenML/ChatTTS-Forge/issues/49) add testcase [[d982e33](https://github.com/lenML/ChatTTS-Forge/commit/d982e33ed30749d7ae6570ade5ec7b560a3d1f06)]
- 🐛 fix Box() index [#49](https://github.com/lenML/ChatTTS-Forge/issues/49) [[1788318](https://github.com/lenML/ChatTTS-Forge/commit/1788318a96c014a53ee41c4db7d60fdd4b15cfca)]
- 🐛 fix &#x60;--use_cpu&#x60; [#47](https://github.com/lenML/ChatTTS-Forge/issues/47) update conftest [[4095b08](https://github.com/lenML/ChatTTS-Forge/commit/4095b085c4c6523f2579e00edfb1569d65608ca2)]
- 🐛 fix &#x60;--use_cpu&#x60; [#47](https://github.com/lenML/ChatTTS-Forge/issues/47) [[221962f](https://github.com/lenML/ChatTTS-Forge/commit/221962fd0f61d3f269918b26a814cbcd5aabd1f0)]
- 🐛 fix webui speaker args [[3b3c331](https://github.com/lenML/ChatTTS-Forge/commit/3b3c3311dd0add0e567179fc38223a3cc5e56f6e)]
- 🐛 fix speaker trainer [[52d473f](https://github.com/lenML/ChatTTS-Forge/commit/52d473f37f6a3950d4c8738c294f048f11198776)]
- 🐛 兼容 win32 [[7ffa37f](https://github.com/lenML/ChatTTS-Forge/commit/7ffa37f3d36fb9ba53ab051b2fce6229920b1208)]
- 🐛 fix google api ssml synthesize [#43](https://github.com/lenML/ChatTTS-Forge/issues/43) [[1566f88](https://github.com/lenML/ChatTTS-Forge/commit/1566f8891c22d63681d756deba70374e2b75d078)]

### Miscellaneous

- Merge pull request [#58](https://github.com/lenML/ChatTTS-Forge/issues/58) from lenML/dependabot/pip/urllib3-2.2.2 [[f259f18](https://github.com/lenML/ChatTTS-Forge/commit/f259f180af57f9a6938b14bf263d0387b6900e57)]
- 📝 update changelog [[b9da7ec](https://github.com/lenML/ChatTTS-Forge/commit/b9da7ec1afed416a825e9e4a507b8263f69bf47e)]
- 📝 update [[8439437](https://github.com/lenML/ChatTTS-Forge/commit/84394373de66b81a9f7f70ef8484254190e292ab)]
- 📝 update [[ef97206](https://github.com/lenML/ChatTTS-Forge/commit/ef972066558d0b229d6d0b3d83bb4f8e8517558f)]
- 📝 improve readme.md [[7bf3de2](https://github.com/lenML/ChatTTS-Forge/commit/7bf3de2afb41b9a29071bec18ee6306ce8e70183)]
- 📝 add bug report forms [[091cf09](https://github.com/lenML/ChatTTS-Forge/commit/091cf0958a719236c77107acf4cfb8c0ba090946)]
- 📝 update changelog [[3d519ec](https://github.com/lenML/ChatTTS-Forge/commit/3d519ec8a20098c2de62631ae586f39053dd89a5)]
- 📝 update [[66963f8](https://github.com/lenML/ChatTTS-Forge/commit/66963f8ff8f29c298de64cd4a54913b1d3e29a6a)]
- 📝 update [[b7a63b5](https://github.com/lenML/ChatTTS-Forge/commit/b7a63b59132d2c8dbb4ad2e15bd23713f00f0084)]

<a name="0.6.0"></a>

## 0.6.0 (2024-06-12)

### Added

- ✨ add XTTSv2 api [#42](https://github.com/lenML/ChatTTS-Forge/issues/42) [[d1fc63c](https://github.com/lenML/ChatTTS-Forge/commit/d1fc63cd1e847d622135c96371bbfe2868a80c19)]
- ✨ google api 支持 enhancer [[14fecdb](https://github.com/lenML/ChatTTS-Forge/commit/14fecdb8ea0f9a5d872a4c7ca862e901990076c0)]
- ✨ 修改 podcast 脚本默认 style [[98186c2](https://github.com/lenML/ChatTTS-Forge/commit/98186c25743cbfa24ca7d41336d4ec84aa34aacf)]
- ✨ playground google api [[4109adb](https://github.com/lenML/ChatTTS-Forge/commit/4109adb317be215970d756b4ba7064c9dc4d6fdc)]
- ✨ 添加 unload api [[ed9d61a](https://github.com/lenML/ChatTTS-Forge/commit/ed9d61a2fe4ba1d902d91517148f8f7dea47b51b)]
- ✨ support api workers [[babdada](https://github.com/lenML/ChatTTS-Forge/commit/babdada50e79e425bac4d3074f8e42dfb4c4c33a)]
- ✨ add ffmpeg version to webui footer [[e9241a1](https://github.com/lenML/ChatTTS-Forge/commit/e9241a1a8d1f5840ae6259e46020684ba70a0efb)]
- ✨ support use internal ffmpeg [[0e02ab0](https://github.com/lenML/ChatTTS-Forge/commit/0e02ab0f5d81fbfb6166793cb4f6d58c5f17f34c)]
- ✨ 增加参数 debug_generate [[94e876a](https://github.com/lenML/ChatTTS-Forge/commit/94e876ae3819c3efbde4a239085f91342874bd5a)]
- ✨ 支持 api 服务与 webui 并存 [[4901491](https://github.com/lenML/ChatTTS-Forge/commit/4901491eced3955c51030388d1dcebf049cd790e)]
- ✨ refiner api support normalize [[ef665da](https://github.com/lenML/ChatTTS-Forge/commit/ef665dad5a5517c610f0b430bc52a5b0ba3c2d96)]
- ✨ add webui 音色编辑器 [[fb4c7b3](https://github.com/lenML/ChatTTS-Forge/commit/fb4c7b3b0949ac669da0d069c739934f116b83e2)]
- ✨ add localization [[c05035d](https://github.com/lenML/ChatTTS-Forge/commit/c05035d5cdcc5aa7efd995fe42f6a2541abe718b)]
- ✨ SSML 支持 enhancer [[5c2788e](https://github.com/lenML/ChatTTS-Forge/commit/5c2788e04f3debfa8bafd8a2e2371dde30f38d4d)]
- ✨ webui 增加 podcast 工具 tab [[b0b169d](https://github.com/lenML/ChatTTS-Forge/commit/b0b169d8b49c8e013209e59d1f8b637382d8b997)]
- ✨ 完善 enhancer [[205ebeb](https://github.com/lenML/ChatTTS-Forge/commit/205ebebeb7530c81fde7ea96c7e4c6a888a29835)]

### Changed

- ⚡ improve synthesize_audio [[759adc2](https://github.com/lenML/ChatTTS-Forge/commit/759adc2ead1da8395df62ea1724456dad6894eb1)]
- ⚡ reduce enhancer chunk vram usage [[3464b42](https://github.com/lenML/ChatTTS-Forge/commit/3464b427b14878ee11e03ebdfb91efee1550de59)]
- ⚡ 增加默认说话人 [[d702ad5](https://github.com/lenML/ChatTTS-Forge/commit/d702ad5ad585978f8650284ab99238571dbd163b)]
- 🍱 add &#x60;podcast&#x60; &#x60;podcast_p&#x60; style [[2b9e5bf](https://github.com/lenML/ChatTTS-Forge/commit/2b9e5bfd8fe4700f802097b995f5b68bf1097087)]
- 🎨 improve code [[317951e](https://github.com/lenML/ChatTTS-Forge/commit/317951e431b16c735df31187b1af7230a1608c41)]
- 🍱 update banner [[dbc293e](https://github.com/lenML/ChatTTS-Forge/commit/dbc293e1a7dec35f60020dcaf783ba3b7c734bfa)]
- ⚡ 增强 TN [[092c1b9](https://github.com/lenML/ChatTTS-Forge/commit/092c1b94147249880198fe2ad3dfe3b209099e19)]
- ⚡ enhancer 支持 off_tqdm [[94d34d6](https://github.com/lenML/ChatTTS-Forge/commit/94d34d657fa3433dae9ff61775e0c364a6f77aff)]
- ⚡ 增加 git env [[43d9c65](https://github.com/lenML/ChatTTS-Forge/commit/43d9c65877ff68ad94716bc2e505ccc7ae8869a8)]
- ⚡ 修改 webui 保存文件格式 [[2da41c9](https://github.com/lenML/ChatTTS-Forge/commit/2da41c90aa81bf87403598aefaea3e0ae2e83d79)]

### Breaking changes

- 💥 enhancer support --half [[fef2ed6](https://github.com/lenML/ChatTTS-Forge/commit/fef2ed659fd7fe5a14807d286c209904875ce594)]

### Removed

- 🔥 clear code [[e8a1d7b](https://github.com/lenML/ChatTTS-Forge/commit/e8a1d7b269d259adc3082a0349c9b73fef1ec1a4)]

### Fixed

- 🐛 fix worker env loader [[5b0bf4e](https://github.com/lenML/ChatTTS-Forge/commit/5b0bf4e93738bcd115f006376691c4eaa89b66de)]
- 🐛 fix colab default lang missing [[d4e5190](https://github.com/lenML/ChatTTS-Forge/commit/d4e51901856305fc039d886a92e38eea2a2cd24d)]
- 🐛 fix &quot;reflection_pad1d&quot; not implemented for &#x27;Half&#x27; [[536c19b](https://github.com/lenML/ChatTTS-Forge/commit/536c19b7f6dc3f1702fcc2a90daa3277040e70f0)]
- 🐛 fix [#33](https://github.com/lenML/ChatTTS-Forge/issues/33) [[76e0b58](https://github.com/lenML/ChatTTS-Forge/commit/76e0b5808ede71ebb28edbf0ce0af7d9da9bcb27)]
- 🐛 fix localization error [[507dbe7](https://github.com/lenML/ChatTTS-Forge/commit/507dbe7a3b92d1419164d24f7804295f6686b439)]
- 🐛 block main thread [#30](https://github.com/lenML/ChatTTS-Forge/issues/30) [[3a7cbde](https://github.com/lenML/ChatTTS-Forge/commit/3a7cbde6ccdfd20a6c53d7625d4e652007367fbf)]
- 🐛 fix webui skip no-translate [[a8d595e](https://github.com/lenML/ChatTTS-Forge/commit/a8d595eb490f23c943d6efc35b65b33266c033b7)]
- 🐛 fix hf.space force abort [[f564536](https://github.com/lenML/ChatTTS-Forge/commit/f5645360dd1f45a7bf112f01c85fb862ee57df3c)]
- 🐛 fix missing device [#25](https://github.com/lenML/ChatTTS-Forge/issues/25) [[07cf6c1](https://github.com/lenML/ChatTTS-Forge/commit/07cf6c1386900999b6c9436debbfcbe59f6b692a)]
- 🐛 fix Chat.refiner_prompt() [[0839863](https://github.com/lenML/ChatTTS-Forge/commit/083986369d0e67fcb4bd71930ad3d2bc3fc038fb)]
- 🐛 fix --language type check [[50d354c](https://github.com/lenML/ChatTTS-Forge/commit/50d354c91c659d9ae16c8eaa0218d9e08275fbb2)]
- 🐛 fix hparams config [#22](https://github.com/lenML/ChatTTS-Forge/issues/22) [[61d9809](https://github.com/lenML/ChatTTS-Forge/commit/61d9809804ad8c141d36afde51a608734a105662)]
- 🐛 fix enhance 下载脚本 [[d2e14b0](https://github.com/lenML/ChatTTS-Forge/commit/d2e14b0a4905724a55b03493fa4b94b5c4383c95)]
- 🐛 fix &#x27;trange&#x27; referenced [[d1a8dae](https://github.com/lenML/ChatTTS-Forge/commit/d1a8daee61e62d14cf5fd7a17fab4424e24b1c41)]
- 🐛 fix ssml to mp3 error &#x60;bad sample width&#x60; [[564b7eb](https://github.com/lenML/ChatTTS-Forge/commit/564b7ebbd55df50aac38562957eebd898945315e)]
- 🐛 fix seed context exit behavior [[d4e33c8](https://github.com/lenML/ChatTTS-Forge/commit/d4e33c8f0aabe253ce96756f907e979578c81b17)]
- 🐛 fix colab script [[687cc2c](https://github.com/lenML/ChatTTS-Forge/commit/687cc2cc97ff7e89328b747dbfcacbcd51bd5efc)]

### Miscellaneous

- 🐳 fix docker / 兼容 py 3.9 [[ebb096f](https://github.com/lenML/ChatTTS-Forge/commit/ebb096f9b1b843b65d150fb34da7d3b5acb13011)]
- 🐳 add .dockerignore [[57262b8](https://github.com/lenML/ChatTTS-Forge/commit/57262b81a8df3ed26ca5da5e264d5dca7b022471)]
- 🧪 add tests [[a807640](https://github.com/lenML/ChatTTS-Forge/commit/a80764030b790baee45a10cbe2d4edd7f183ef3c)]
- 🌐 fix [[b34a0f8](https://github.com/lenML/ChatTTS-Forge/commit/b34a0f8654467f3068e43056708742ab69e3665b)]
- 🌐 remove chat limit desc [[3f81eca](https://github.com/lenML/ChatTTS-Forge/commit/3f81ecae6e4521eeb4e867534defc36be741e1e2)]
- 🧪 add tests [[7a54225](https://github.com/lenML/ChatTTS-Forge/commit/7a542256a157a281a15312bbf987bc9fb16876ee)]
- 🔨 improve model downloader [[79a0c59](https://github.com/lenML/ChatTTS-Forge/commit/79a0c599f03b4e47346315a03f1df3d92578fe5d)]
- 🌐 更新翻译文案 [[f56caa7](https://github.com/lenML/ChatTTS-Forge/commit/f56caa71e9186680b93c487d9645186ae18c1dc6)]

<a name="0.5.5"></a>

## 0.5.5 (2024-06-08)

### Added

- ✨ add webui speaker creator [[df26549](https://github.com/lenML/ChatTTS-Forge/commit/df265490f35b2b991c395455dd2f4ad563193cef)]
- ✨ webui speaker tab and merger [[7ad71fd](https://github.com/lenML/ChatTTS-Forge/commit/7ad71fddb61f3b41b3af66d201f6105ca09539d9)]
- ✨ add enhance download script [[37adec6](https://github.com/lenML/ChatTTS-Forge/commit/37adec6de3109b3829602c7c7be06fd7247f10eb)]
- ✨ add audio enhance/denoise [[00cbc8e](https://github.com/lenML/ChatTTS-Forge/commit/00cbc8e96833fbcaf6cc224dc330908fa647f317)]
- ✅ add speakers api test [[fbe4304](https://github.com/lenML/ChatTTS-Forge/commit/fbe4304c6716fb182442d356dbe3976982ca9d2b)]
- ✅ add unit test [[e7f9385](https://github.com/lenML/ChatTTS-Forge/commit/e7f938562c1173899cc4e7330d59a8e354cafea4)]

### Changed

- ♿ pin resemble-enhance [[b8f41f9](https://github.com/lenML/ChatTTS-Forge/commit/b8f41f90061c75ee3e09ddc6cae8d657bc67aad1)]
- ⚡ 调整 speaker 合并 step [[906ecc3](https://github.com/lenML/ChatTTS-Forge/commit/906ecc3d295d90459485cd131563ffd588914d52)]
- ♻️ SSML refactor [[6666082](https://github.com/lenML/ChatTTS-Forge/commit/6666082375c43b143d242bf425053e2ae661eb09)]
- ♻️ webui refactor [[7585282](https://github.com/lenML/ChatTTS-Forge/commit/75852822f7d9cd8b95b557e2870e5435a0932fa1)]
- ⚡ add benchmark [[ddb7670](https://github.com/lenML/ChatTTS-Forge/commit/ddb76704e5e6847bb0eeca2c0b50764a66783686)]

### Removed

- 🔥 remove trainer [[0c80c24](https://github.com/lenML/ChatTTS-Forge/commit/0c80c2437fb8e8b231ae770205089198f4ac1c13)]

### Fixed

- 🐛 fix warning and hf.spaces error [[f9700bb](https://github.com/lenML/ChatTTS-Forge/commit/f9700bbb1b057b2dfe4437de7cbd41a659be76c5)]
- 🐛 fix model thread competition [[0ade6ac](https://github.com/lenML/ChatTTS-Forge/commit/0ade6ac07a2c75eb1cdda1c3db8bdf9bc2665244)]
- 🐛 fix hf space error &#x60;ZeroGPU has not been initialized&#x60; [[562e17c](https://github.com/lenML/ChatTTS-Forge/commit/562e17c9372278c03705ec5a3ec77750854d5c7e)]
- 🐛 fix openai api [[49088c5](https://github.com/lenML/ChatTTS-Forge/commit/49088c5480043518bb9beda817f5e5b38d133fa8)]

### Miscellaneous

- 🐳 fix pip requirements [[4256371](https://github.com/lenML/ChatTTS-Forge/commit/4256371c9d3d8d290840a98fb6ac7bc19268a1e7)]

<a name="0.5.2"></a>

## 0.5.2 (2024-06-06)

### Changed

- ⚡ improve TN [[6744323](https://github.com/lenML/ChatTTS-Forge/commit/6744323df814430b2d92c3f16329ab8f09eb4ad3)]

### Fixed

- 🐛 fix window proxy env [[d0f9760](https://github.com/lenML/ChatTTS-Forge/commit/d0f97608cef2afdbeb803c906a71e05dbf2424a1)]

### Miscellaneous

- 📝 add banchmark [[3a72ba0](https://github.com/lenML/ChatTTS-Forge/commit/3a72ba0f97d5409502b9ff98e356f69affcce06b)]

<a name="0.5.1"></a>

## 0.5.1 (2024-06-06)

### Fixed

- 🐛 fix SynthesizeSegments seed [[83b63bd](https://github.com/lenML/ChatTTS-Forge/commit/83b63bdd0d92e115c9b6946f427343c48de1a313)]

<a name="0.5.0"></a>

## 0.5.0 (2024-06-06)

### Added

- ✨ add systeam versions info [[ff94763](https://github.com/lenML/ChatTTS-Forge/commit/ff947636c5e69d6bdf5111f95d8afb979d157fba)]
- ✨ torch_gc [[d8a8f35](https://github.com/lenML/ChatTTS-Forge/commit/d8a8f35958c25d931ce47b53730d388a71e86b2d)]
- ✨ normalize improve [[c9db440](https://github.com/lenML/ChatTTS-Forge/commit/c9db440b2719119285c6536c2c4658afdb20ff27)]
- ✨ improve playground speaker manager [[01ebda3](https://github.com/lenML/ChatTTS-Forge/commit/01ebda3a28cfc5f1e78f8a434b1077f01b22f399)]
- ✨ improve speaker manager [[1b377d4](https://github.com/lenML/ChatTTS-Forge/commit/1b377d448214e232d477bd828fca5eba6aa87e7b)]
- ✨ add speakers [[6c4aa29](https://github.com/lenML/ChatTTS-Forge/commit/6c4aa29f147942d1c2f3c095b4d832409e53e5cb)]
- ✨ playground preact -&gt; react [[450a0f9](https://github.com/lenML/ChatTTS-Forge/commit/450a0f9d184c8c5d8df6cce3d8e4596c543dbfe1)]
- ✨ batch_size in api [[616a262](https://github.com/lenML/ChatTTS-Forge/commit/616a262012d1e23dd877bbfc4cde4f16f477d1a7)]
- ✨ add .env file [[a0eddee](https://github.com/lenML/ChatTTS-Forge/commit/a0eddeefc141630d43496881e73d5fcd90742828)]
- ✨ improve sentence spliter for markdown [[7cac79a](https://github.com/lenML/ChatTTS-Forge/commit/7cac79ad6720c98bdec0903dd473de70bdcef137)]
- ✨ support batch generate [[cb9d9aa](https://github.com/lenML/ChatTTS-Forge/commit/cb9d9aa5d55482a51018a4ebe95d84d95d803cc4)]
- ✨ add cli args, off_tqdm / half [[dff2098](https://github.com/lenML/ChatTTS-Forge/commit/dff2098b76deac30d738ce05ba99225914b781a3)]
- ✨ add cli args, no_playground no_docs [[98629a2](https://github.com/lenML/ChatTTS-Forge/commit/98629a2c46a376a3955ea8862c9b80361f44f8a3)]
- ✨ colab script [[5943fd9](https://github.com/lenML/ChatTTS-Forge/commit/5943fd9c168ad98c0d2caf6deffb1d63848d75c3)]
- ✨ emoji normalize [[c88fc3f](https://github.com/lenML/ChatTTS-Forge/commit/c88fc3f18f0ae512f477880cfea972376f825ab7)]
- ✨ improve webui [[c48c227](https://github.com/lenML/ChatTTS-Forge/commit/c48c2278d8a123694ecf9deeb36ea4d5e3a0499e)]
- ✨ add download models script [[581f278](https://github.com/lenML/ChatTTS-Forge/commit/581f27859dad4a6c3d1b4af9ebc854fc2f4829df)]
- ✨ add spks [[d2a7364](https://github.com/lenML/ChatTTS-Forge/commit/d2a736485907490e1dbf0db334ec2b6459d35dbe)]
- ✨ add dockerfile [[fc0f4e7](https://github.com/lenML/ChatTTS-Forge/commit/fc0f4e73f7f8e1698afd97065a84161f7807e655)]
- ✨ add ssml example / fix ssml [[a2c18b1](https://github.com/lenML/ChatTTS-Forge/commit/a2c18b149584137d323d14f0923a6c2f98969e89)]
- ✨ add styles [[e106b1b](https://github.com/lenML/ChatTTS-Forge/commit/e106b1bc3edffd71c8da8408e57444b50c0cc91a)]
- ✨ webui [[3c959ad](https://github.com/lenML/ChatTTS-Forge/commit/3c959ad341775e335233e6dd6797bca045ec106e)]
- 🎉 base code all in one [[3051588](https://github.com/lenML/ChatTTS-Forge/commit/30515881b31a5bb7b08be273f4cdb3c9a9854a6c)]

### Changed

- ⚡ docker [[6b8eed1](https://github.com/lenML/ChatTTS-Forge/commit/6b8eed1caa69af9787a019fb226d25a155ca58af)]
- ⚡ playground improve [[f8da40b](https://github.com/lenML/ChatTTS-Forge/commit/f8da40b773d4485f69dfd207fc33294dfac61b93)]
- ⚡ improve normalize for EN [[3717ae3](https://github.com/lenML/ChatTTS-Forge/commit/3717ae31478f561f9dd83fad8983282e7f9380c9)]
- ⚡ dockerfile [[8ad7659](https://github.com/lenML/ChatTTS-Forge/commit/8ad7659f62af5436545da239ee8f8f7cb3d6c103)]
- ⚡ improve webui [[4ac24e8](https://github.com/lenML/ChatTTS-Forge/commit/4ac24e835ca078ca09f81e41980757a84cc4fccd)]
- ⚡ improve [[153fa4f](https://github.com/lenML/ChatTTS-Forge/commit/153fa4f152b3ed084cb2b5e6f0b12d86f40f77f3)]
- ⚡ improve [[2dbc76d](https://github.com/lenML/ChatTTS-Forge/commit/2dbc76ddc9937d2df63106f77197fa28a5b0d23b)]

### Fixed

- 🐛 fix playground package url [[cc33013](https://github.com/lenML/ChatTTS-Forge/commit/cc33013a5a0a4c7f21011e44dbc0fb29228482fa)]
- 🐛 fix webui TN [[9f1e8f4](https://github.com/lenML/ChatTTS-Forge/commit/9f1e8f45dce8f519526896df7033fbc40690abeb)]
- 🐛 improve rng [#11](https://github.com/lenML/ChatTTS-Forge/issues/11) [[af06646](https://github.com/lenML/ChatTTS-Forge/commit/af06646877f56ea458a8b316dca4160d546bb4b5)]
- 🐛 fix webui segment limit [[a74034f](https://github.com/lenML/ChatTTS-Forge/commit/a74034f0fb86a997d621605967ceca59351f2627)]
- 🐛 fix speaker hash [[bd3e532](https://github.com/lenML/ChatTTS-Forge/commit/bd3e5324cf260ce75997a2b5d087af7d0c0106fe)]
- 🐛 修复 openai api speed 验证 [[124a430](https://github.com/lenML/ChatTTS-Forge/commit/124a4309e7be0e146379e175eb9cde6374ea207d)]
- 🐛 fix 接口地址调整 + 文档 [#9](https://github.com/lenML/ChatTTS-Forge/issues/9) [[3bf349e](https://github.com/lenML/ChatTTS-Forge/commit/3bf349e274d8ab343688d2d76bd59edd29c742ed)]
- 🐛 fix docker python env &#x60;unsupported operand type(s)&#x60; [[c2c9658](https://github.com/lenML/ChatTTS-Forge/commit/c2c965890334736ae0887c214fd731426239c3e8)]
- 🐛 fix webui ssml [[f2722e3](https://github.com/lenML/ChatTTS-Forge/commit/f2722e38a00cfd03cb44589d30148ecea6703cd5)]
- 🐛 fix batch window rolling [[63434b3](https://github.com/lenML/ChatTTS-Forge/commit/63434b34ab35d77dd91e6d6c244477459a8bf44f)]
- 🐛 fix colab error [[61121e9](https://github.com/lenML/ChatTTS-Forge/commit/61121e9c7150d64d5a3112e9e266eecc764c3e63)]
- 🐛 find speaker by name [[25610b8](https://github.com/lenML/ChatTTS-Forge/commit/25610b877c12bf68fd16064489c7d24354a1d900)]
- 🐛 fix dropdown miss [[e5eab54](https://github.com/lenML/ChatTTS-Forge/commit/e5eab54454069a4f7575ceee29a36960b916e015)]
- 🐛 fix env read [[f9cb9d0](https://github.com/lenML/ChatTTS-Forge/commit/f9cb9d099bd022635fe078155598411fdd7df00e)]
- 🐛 fix env loader [[d724659](https://github.com/lenML/ChatTTS-Forge/commit/d724659389197637f152b09079a36f6b1f26e79b)]
- 🐛 fix colab no half [[dce20c8](https://github.com/lenML/ChatTTS-Forge/commit/dce20c8e1592cce2871953a42f7ebfa7b0f0f54e)]
- 🐛 fix tqdm referenced [[c29fd5c](https://github.com/lenML/ChatTTS-Forge/commit/c29fd5ca82923fa1be0de70137f4ce0e2e32f4ae)]
- 🐛 fix [#6](https://github.com/lenML/ChatTTS-Forge/issues/6) [[fc30977](https://github.com/lenML/ChatTTS-Forge/commit/fc309774a73c3c9ae4c5c527c3aa9bef4a7a3c1f)]
- 🐛 fix seed context error [[faceb2b](https://github.com/lenML/ChatTTS-Forge/commit/faceb2b0af749a869df3656dee89b20f69b130b8)]
- 🐛 fix infer_seed range [[2782182](https://github.com/lenML/ChatTTS-Forge/commit/2782182367faef932429e2a7a012f7d867c0cb3a)]
- 🐛 fix [#2](https://github.com/lenML/ChatTTS-Forge/issues/2) 改用境内 cdn [[6ba27c3](https://github.com/lenML/ChatTTS-Forge/commit/6ba27c3c4d1ef4175cfe95de671599f64572eaeb)]
- 🐛 fix [#3](https://github.com/lenML/ChatTTS-Forge/issues/3) [[7ffe91f](https://github.com/lenML/ChatTTS-Forge/commit/7ffe91fa3028f88f0414786dd669b91cce409043)]
- 🐛 fix webui [[b44156f](https://github.com/lenML/ChatTTS-Forge/commit/b44156fbac7ea59765ced4bbfd303f3095166693)]
- 🐛 fix webui [[4ad0cee](https://github.com/lenML/ChatTTS-Forge/commit/4ad0ceef9d1251437cff57e12a0a82f1cd427e17)]
- 🐛 适配大写 % [[5de4bf7](https://github.com/lenML/ChatTTS-Forge/commit/5de4bf7e19d436c93bc584af859fc5e58596f0c9)]
- 🐛 webui normalize [[d8113f8](https://github.com/lenML/ChatTTS-Forge/commit/d8113f8c336d4a1b8e8fb0e82d0ba2ed02a36086)]
- 🐛 webui spk style fix [[3319358](https://github.com/lenML/ChatTTS-Forge/commit/3319358a0c51930cca6cc6c9326a694aff15c8c4)]
- 🐛 speaker load [[588848d](https://github.com/lenML/ChatTTS-Forge/commit/588848d30fa4e3fc669ced81d780517145be3e2e)]
- 🐛 speaker load [[3f63aa7](https://github.com/lenML/ChatTTS-Forge/commit/3f63aa76911274ddf62c47298d1758b0fd984e52)]
- 🐛 playground base_url [[8b468fc](https://github.com/lenML/ChatTTS-Forge/commit/8b468fc67e6c4b649844a419b8dda77bf01ea439)]

### Miscellaneous

- :zep: improve cache [[eedc558](https://github.com/lenML/ChatTTS-Forge/commit/eedc55880f4802ef9f23117d8266100be18cbe41)]
- :zep: Improved device support [[400afe6](https://github.com/lenML/ChatTTS-Forge/commit/400afe60c9261247998617b2943fa210230b7ad3)]
- :zep: improve TN [[c22591f](https://github.com/lenML/ChatTTS-Forge/commit/c22591ffb2c1ae9e14eef307d708925bcc7a67a7)]
- add mps support [[6983506](https://github.com/lenML/ChatTTS-Forge/commit/6983506d7237c05ea346fdbe7d042b69d3743a3b)]
- :zep: revert infer_utils [[4a507cc](https://github.com/lenML/ChatTTS-Forge/commit/4a507cccf23b94a11f470b67d96bbad6c9efbf74)]
- Optimize tqdm display [[516eca6](https://github.com/lenML/ChatTTS-Forge/commit/516eca6c99b7b8433d1cda8435a19214eb1cc678)]
- Improve code [[0e278ab](https://github.com/lenML/ChatTTS-Forge/commit/0e278ab40276e3e0610515bf0e4329c95a67e00c)]
- Add apple gpu mps backend [[9764e23](https://github.com/lenML/ChatTTS-Forge/commit/9764e233e1e82d8dbec607032744cb565b616525)]
- Update issue templates [[1d19417](https://github.com/lenML/ChatTTS-Forge/commit/1d194177ab6dd3b1055ca925bfd2befc6fe2f07c)]
- LICENSE [[0afb189](https://github.com/lenML/ChatTTS-Forge/commit/0afb189718797e0706ba64c1e8d2188c4fa0fe4c)]
- Initial commit [[9fed2b6](https://github.com/lenML/ChatTTS-Forge/commit/9fed2b60a90547286e6c06483167c397c7bbed78)]
