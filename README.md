## here is the link for the video: 
https://vanderbilt.app.box.com/file/2217083363044?s=e5shuujs43kur3zfuirsnknjq6pn9801
/n

https://vanderbilt.box.com/s/p1slvswmyji1zgjm4c27uofeamyxvgno

# AI-Assisted AR Medical Image Exploration using Multimodal Models

## Latest Prototype Media

The GIF below shows the current mobile AR prototype running on a Samsung Galaxy phone. It demonstrates on-device placement of the anatomy model in a real outdoor scene rather than a desktop-only mockup.

![Samsung Galaxy mobile AR demo showing anatomy placement in a real scene](galxyscreen.gif)

## Current Prototype

The current project combines two practical prototype paths:

- a native Python workflow built with `SimpleITK`, `VTK`, and `PyVista` for preprocessing, registration, and 3D visualization
- a lightweight Unity Android AR workflow built with `AR Foundation` and `ARCore` for on-device placement and interaction

The current prototype code is stored in:

- `native_vtk_prototype/app`
- `native_vtk_prototype/scripts`

Earlier screenshot from the initial chatbot-style concept:

![ARMedVLM chatbot-style prototype screenshot](image.png)

The main prototype capabilities are:

- cleaned 3D CT rendering with reduced table and background artifacts
- rigid registration of longitudinal scans into a fixed reference (`CT4`)
- playback of registered series (`CT1`, `CT3`, `CT4`)
- mobile AR placement of anatomy content in a real scene
- simple phone-based rotation and scaling interaction

## Android AR Prototype

The Unity Android branch was used as a practical hardware validation step for phone deployment.

Current Android prototype scope:

- detect horizontal planes with `ARCore`
- place the active anatomy object onto a real-world surface
- preserve the existing CT selection flow (`CT1` to `CT4`)
- preserve the mode flow (`Bone`, `Bone+S`, `Organs`)
- use one-finger drag for rotation
- use two-finger pinch for scale

## Hardware Validation

The mobile progression used two practical hardware checkpoints:

- `Galaxy S8`: lightweight sanity-check build for safe mobile startup and incremental validation
- `Galaxy S20`: ARCore-enabled prototype path for plane detection and real-world anatomy placement

The top GIF reflects the phone-based AR direction directly: it shows the anatomy anchored into a live camera scene on Android hardware.

The current processing flow is:

```text
Raw CT -> body cleanup -> spleen mask alignment -> rigid registration to CT4 -> 3D viewer / timeline / dashboard
```

Typical local run order for the prototype is:

```powershell
cd C:\Users\adams\Documents\Projects\ARMedVLMProposal\native_vtk_prototype
.\scripts\run_preprocess_all.ps1
.\scripts\run_register_all.ps1
.\scripts\run_timeline.ps1
```

For the interactive native dashboard:

```powershell
cd C:\Users\adams\Documents\Projects\ARMedVLMProposal\native_vtk_prototype
.\scripts\run_native_dashboard.ps1
```

## Project Goal

This project explores an augmented reality workflow for volumetric medical images such as CT scans. The goal is to combine 3D visualization, interactive manipulation, and multimodal interpretation in a single workflow instead of treating them as separate tools.

## Technical Approach

The system has four main components: data preprocessing, AR visualization, interaction, and AI inference.

### 1. Data Processing

Input data will consist of CT volumes in NIfTI or DICOM format. The preprocessing stage will handle resampling, intensity normalization, and optional segmentation or threshold-based extraction of relevant structures. The processed data will then be converted into either:

- slice textures for 2D or pseudo-volume interaction inside AR
- surface meshes generated from thresholding or marching cubes for 3D rendering

This stage is implemented in Python using libraries such as `SimpleITK` and `VTK`.

### 2. AR Visualization

The visualization environment is planned around `Unity`, using `AR Foundation` as the main cross-platform AR framework. Depending on available hardware, the deployment target can be:

- iPhone or iPad using `ARKit`
- Android device using `ARCore`
- Meta Quest in VR or passthrough mode if AR deployment becomes impractical

The rendering system will support:

- display of a 3D medical object or stack of slices
- slicing plane interaction
- threshold or opacity adjustments
- camera-relative repositioning and scaling of the medical volume

### 3. Interaction Layer

The user can:

- rotate and scale the volume
- move a slicing plane through the scan
- select or define a region of interest
- request semantic interpretation of the currently selected slice or region

Interaction is implemented with standard Unity UI controls and touch-first mobile input.

### 4. AI Inference

The AI component uses a biomedical multimodal model such as `BiomedCLIP` or `MedSigLIP` for image-text alignment. A selected slice or ROI is passed to a Python inference pipeline, and the resulting semantic output is returned to the interface.

The expected inference path is:

`CT volume -> selected slice or ROI -> multimodal model -> similarity scores or predicted concept -> AR overlay or UI text`

### System Architecture

```text
Medical Image Volume
        |
        v
Preprocessing Pipeline
        |
        v
Unity AR Application
|-------------------------------|
| Rendering | Interaction | AI  |
|-------------------------------|
        |
        v
Selected Slice / ROI
        |
        v
BiomedCLIP / MedSigLIP Inference
        |
        v
Semantic Output in AR
```

## Contribution

This project is a systems integration effort rather than a new model contribution. Its main value is combining medical image visualization and multimodal interpretation inside a single AR-oriented workflow.

## Evaluation Plan

### 1. End-to-End Functional Evaluation

The first criterion is whether the full pipeline works:

- load a medical volume
- render it correctly in Unity
- allow slice or ROI selection
- send the selected image data to the multimodal model
- display the returned semantic result inside the interface

Success means reliable end-to-end execution across a small set of test volumes.

### 2. Performance Evaluation

The system must remain usable interactively. The following metrics will be measured:

- rendering frame rate
- latency from user interaction to semantic response
- memory usage for typical volumes

Target values:

- interactive rendering near or above 30 FPS
- AI response latency below 1 to 2 seconds for a selected slice

### 3. Comparative Evaluation

To justify the AI component, the project will compare:

- AR visualization only
- AR visualization with semantic model output

The evaluation question is whether the model output adds meaningful information during exploration.

### 4. Stress Testing

The system will also be stress-tested on:

- larger volumes
- noisier scans
- different scan orientations or preprocessing conditions

This helps define practical operating limits and failure modes.

## Milestones and Contingencies

### Minimum Viable Demo

The minimum viable version of the project will include:

- loading a CT volume into Unity
- basic volume placement and slicing interaction
- export of a selected slice or ROI
- inference with a biomedical image-text model
- display of the returned semantic result in the Unity interface

### Expected Hardest Component

The hardest component is likely to be the integration between Unity and the AI inference pipeline, especially if low-latency interaction is required.

### Contingency Plan

If full AR deployment becomes too difficult within the available time, the fallback will be:

- switch from mobile AR to desktop Unity or VR mode
- use preprocessed PNG slices instead of full volumetric data
- precompute model outputs for a set of slices rather than performing fully live inference

This keeps the project technically valid while reducing engineering risk.

## Success Criteria

The project will be considered successful if it satisfies the following:

- the system runs end-to-end on at least one medical volume
- the user can manipulate the medical data interactively
- the multimodal model returns meaningful semantic outputs for selected slices or ROIs
- the interface remains responsive enough for live demonstration

## Repository Scope

This repository is intended to store the project proposal, technical design notes, implementation code, and future evaluation artifacts for the AR medical image exploration system.
