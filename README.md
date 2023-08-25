# ReshadeEffectShaderToggler-BG3
REST config for Baldur's Gate 3

## Requirements
* [REST v1.2.7](https://github.com/4lex4nder/ReshadeEffectShaderToggler/releases/tag/v1.2.7)
* ReShade >= 5.9.1
* Baldur's Gate 3 running in DX11 mode
* Have the "Subsurface Scattering" Video option enabled

## What does it do
* Applies effects before the UI is rendered by default. 
* There is a group for applying effects before the game renders things like fog and water.
* If you're using DLSS, effects are applied before the game upscales the image, resulting in a performance boost for some heavier effects.
* Effects making using of known formats reading motion vector and normal data will see a benefit of having the game's resources accessible directly (Launchpad/DRME style formats provided by `bg3_crashpad.fx`).

## What does it not do
Provide a perfect solution. Since effects are, by default, applied onto a render target that has yet to be clamped, many effects are simply going to produce garbage colors.

## Batteries included
Some example effects using game data provided by REST

### bg3_common.fxh
If you know your way around shaders, you can use this header file to access some game engine data in your ReShade effects. Including:
* Perspective projection matrices
* Normal buffer
* Albedo

### bg3_xegtao.fx
A port of [Intel's XeGTAO](https://github.com/GameTechDev/XeGTAO) AO shader to ReShade making full use of `bg3_common.fxh`. Applied before fog by default.

### bg3_xegtao_gi.fx
My poor, barebones attempt at using [visibility bitmasks](https://arxiv.org/abs/2301.11376) hacked into XeGTAO for AO and GI. Requires `bg3_crashpad.fx` running before it. Applied before fog by default.

### bg3_crashpad.fx
A helper shader providing the game's normals and motion vectors in known formats. Executed before fog by default.
