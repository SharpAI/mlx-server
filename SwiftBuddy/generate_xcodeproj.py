#!/usr/bin/env python3
"""
generate_xcodeproj.py — Generates SwiftBuddy.xcodeproj without xcodegen.
Includes MLXInferenceCore sources directly + local SPM packages (mlx-swift, mlx-swift-lm).
Run from the SwiftBuddy/ directory:
    python3 generate_xcodeproj.py
"""

import os, uuid
from pathlib import Path

def uid():
    return uuid.uuid4().hex[:24].upper()

# ── UUIDs ─────────────────────────────────────────────────────────────
PROJ          = uid()
MAIN_GRP      = uid()
SOURCES_GRP   = uid()
CORE_GRP      = uid()
VIEWS_GRP     = uid()
VIEWMODELS_GRP= uid()
PRODUCTS_GRP  = uid()
APP_PRODUCT   = uid()
APP_TARGET    = uid()
PHASE_SRC     = uid()
PHASE_RES     = uid()
PHASE_FWK     = uid()
PROJ_CFGLIST  = uid()
PROJ_DEBUG    = uid()
PROJ_RELEASE  = uid()
TGT_CFGLIST   = uid()
TGT_DEBUG     = uid()
TGT_RELEASE   = uid()

# Local SPM packages
PKG_MLX       = uid()
PKG_MLXLM     = uid()
PKG_TRANSFORMERS = uid()
PKG_HUMMINGBIRD = uid()
PKG_SWIFTSOUP = uid()

# SPM product dependencies
PROD_MLX      = uid()
PROD_MLXLLM   = uid()
PROD_MLXLMC   = uid()
PROD_MLXVLM   = uid()
PROD_HUB      = uid()
PROD_TOKENIZERS = uid()
PROD_HUMMINGBIRD = uid()
PROD_SWIFTSOUP = uid()

# Build files for SPM products (in Frameworks phase)
BF_MLX_FWK    = uid()
BF_MLXLLM_FWK = uid()
BF_MLXLMC_FWK = uid()
BF_MLXVLM_FWK = uid()
BF_HUB_FWK    = uid()
BF_TOKENIZERS_FWK = uid()
BF_HUMMINGBIRD_FWK = uid()
BF_SWIFTSOUP_FWK = uid()

ASSETS_REF    = uid()
ASSETS_BF     = uid()

# ── App source files (relative to SwiftBuddy/)
app_sources = []
for p in sorted(Path("SwiftBuddy").rglob("*.swift")):
    app_sources.append((str(p), uid(), uid()))

# ── MLXInferenceCore sources (path relative to SwiftBuddy/)
core_sources = [
    ("../Sources/MLXInferenceCore/ChatMessage.swift",          uid(), uid()),
    ("../Sources/MLXInferenceCore/GenerationConfig.swift",     uid(), uid()),
    ("../Sources/MLXInferenceCore/ModelCatalog.swift",         uid(), uid()),
    ("../Sources/MLXInferenceCore/ModelStorage.swift",         uid(), uid()),
    ("../Sources/MLXInferenceCore/ModelDownloader.swift",      uid(), uid()),
    ("../Sources/MLXInferenceCore/ModelDownloadManager.swift", uid(), uid()),
    ("../Sources/MLXInferenceCore/HFModelSearch.swift",        uid(), uid()),
    ("../Sources/MLXInferenceCore/ModelArchitectureProbe.swift", uid(), uid()),
    ("../Sources/MLXInferenceCore/InferenceEngine.swift",      uid(), uid()),
    ("../Sources/MLXInferenceCore/SSDStreamingRecovery.swift", uid(), uid()),
]

all_sources = app_sources + core_sources

def pbxproj():
    # PBXBuildFile entries
    build_files = ""
    for path, fref, bf in all_sources:
        name = Path(path).name
        build_files += f"\t\t{bf} /* {name} in Sources */ = {{isa = PBXBuildFile; fileRef = {fref} /* {name} */; }};\n"
    build_files += f"\t\t{ASSETS_BF} /* Assets.xcassets in Resources */ = {{isa = PBXBuildFile; fileRef = {ASSETS_REF} /* Assets.xcassets */; }};\n"
    build_files += f"\t\t{BF_MLX_FWK} /* MLX in Frameworks */ = {{isa = PBXBuildFile; productRef = {PROD_MLX} /* MLX */; }};\n"
    build_files += f"\t\t{BF_MLXLLM_FWK} /* MLXLLM in Frameworks */ = {{isa = PBXBuildFile; productRef = {PROD_MLXLLM} /* MLXLLM */; }};\n"
    build_files += f"\t\t{BF_MLXLMC_FWK} /* MLXLMCommon in Frameworks */ = {{isa = PBXBuildFile; productRef = {PROD_MLXLMC} /* MLXLMCommon */; }};\n"
    build_files += f"\t\t{BF_MLXVLM_FWK} /* MLXVLM in Frameworks */ = {{isa = PBXBuildFile; productRef = {PROD_MLXVLM} /* MLXVLM */; }};\n"
    build_files += f"\t\t{BF_HUB_FWK} /* Hub in Frameworks */ = {{isa = PBXBuildFile; productRef = {PROD_HUB} /* Hub */; }};\n"
    build_files += f"\t\t{BF_TOKENIZERS_FWK} /* Tokenizers in Frameworks */ = {{isa = PBXBuildFile; productRef = {PROD_TOKENIZERS} /* Tokenizers */; }};\n"
    build_files += f"\t\t{BF_HUMMINGBIRD_FWK} /* Hummingbird in Frameworks */ = {{isa = PBXBuildFile; productRef = {PROD_HUMMINGBIRD} /* Hummingbird */; }};\n"
    build_files += f"\t\t{BF_SWIFTSOUP_FWK} /* SwiftSoup in Frameworks */ = {{isa = PBXBuildFile; productRef = {PROD_SWIFTSOUP} /* SwiftSoup */; }};\n"

    # PBXFileReference entries
    file_refs = ""
    for path, fref, _ in app_sources:
        name = Path(path).name
        # SOURCE_ROOT = project root (SwiftBuddy/) so the path is exact, no doubling
        file_refs += f'\t\t{fref} /* {name} */ = {{isa = PBXFileReference; lastKnownFileType = sourcecode.swift; name = "{name}"; path = "{path}"; sourceTree = "SOURCE_ROOT"; }};\n'
    for path, fref, _ in core_sources:
        name = Path(path).name
        # <group> is fine here — these paths start with ../Sources which go up from project dir
        file_refs += f'\t\t{fref} /* {name} */ = {{isa = PBXFileReference; lastKnownFileType = sourcecode.swift; name = "{name}"; path = "{path}"; sourceTree = "<group>"; }};\n'
    file_refs += f'\t\t{ASSETS_REF} /* Assets.xcassets */ = {{isa = PBXFileReference; lastKnownFileType = folder.assetcatalog; path = SwiftBuddy/Assets.xcassets; sourceTree = "SOURCE_ROOT"; }};\n'
    file_refs += f'\t\t{APP_PRODUCT} /* SwiftBuddy.app */ = {{isa = PBXFileReference; explicitFileType = wrapper.application; includeInIndex = 0; path = SwiftBuddy.app; sourceTree = BUILT_PRODUCTS_DIR; }};\n'

    # Group children
    app_root_children = "\n".join(
        f"\t\t\t\t{fref} /* {Path(p).name} */,"
        for p, fref, _ in app_sources if p.count("/") == 1
    )
    views_children = "\n".join(
        f"\t\t\t\t{fref} /* {Path(p).name} */,"
        for p, fref, _ in app_sources if "Views/" in p
    )
    viewmodel_children = "\n".join(
        f"\t\t\t\t{fref} /* {Path(p).name} */,"
        for p, fref, _ in app_sources if "ViewModels/" in p
    )
    core_children = "\n".join(
        f"\t\t\t\t{fref} /* {Path(p).name} */,"
        for p, fref, _ in core_sources
    )

    # Sources build phase (all .swift files)
    src_build_files = "\n".join(
        f"\t\t\t\t{bf} /* {Path(p).name} in Sources */,"
        for p, _, bf in all_sources
    )

    return f"""// !$*UTF8*$!
{{
\tarchiveVersion = 1;
\tclasses = {{
\t}};
\tobjectVersion = 56;
\tobjects = {{

/* Begin PBXBuildFile section */
{build_files}/* End PBXBuildFile section */

/* Begin PBXFileReference section */
{file_refs}/* End PBXFileReference section */

/* Begin PBXFrameworksBuildPhase section */
\t\t{PHASE_FWK} /* Frameworks */ = {{
\t\t\tisa = PBXFrameworksBuildPhase;
\t\t\tbuildActionMask = 2147483647;
\t\t\tfiles = (
\t\t\t\t{BF_MLX_FWK} /* MLX in Frameworks */,
\t\t\t\t{BF_MLXLLM_FWK} /* MLXLLM in Frameworks */,
\t\t\t\t{BF_MLXLMC_FWK} /* MLXLMCommon in Frameworks */,
\t\t\t\t{BF_MLXVLM_FWK} /* MLXVLM in Frameworks */,
\t\t\t\t{BF_HUB_FWK} /* Hub in Frameworks */,
\t\t\t\t{BF_TOKENIZERS_FWK} /* Tokenizers in Frameworks */,
\t\t\t\t{BF_HUMMINGBIRD_FWK} /* Hummingbird in Frameworks */,
\t\t\t\t{BF_SWIFTSOUP_FWK} /* SwiftSoup in Frameworks */,
\t\t\t);
\t\t\trunOnlyForDeploymentPostprocessing = 0;
\t\t}};
/* End PBXFrameworksBuildPhase section */

/* Begin PBXGroup section */
\t\t{MAIN_GRP} = {{
\t\t\tisa = PBXGroup;
\t\t\tchildren = (
\t\t\t\t{CORE_GRP} /* MLXInferenceCore */,
\t\t\t\t{SOURCES_GRP} /* SwiftBuddy */,
\t\t\t\t{PRODUCTS_GRP} /* Products */,
\t\t\t);
\t\t\tsourceTree = "<group>";
\t\t}};
\t\t{PRODUCTS_GRP} /* Products */ = {{
\t\t\tisa = PBXGroup;
\t\t\tchildren = (
\t\t\t\t{APP_PRODUCT} /* SwiftBuddy.app */,
\t\t\t);
\t\t\tname = Products;
\t\t\tsourceTree = "<group>";
\t\t}};
\t\t{CORE_GRP} /* MLXInferenceCore */ = {{
\t\t\tisa = PBXGroup;
\t\t\tchildren = (
{core_children}
\t\t\t);
\t\t\tname = MLXInferenceCore;
\t\t\tsourceTree = "<group>";
\t\t}};
\t\t{SOURCES_GRP} /* SwiftBuddy */ = {{
\t\t\tisa = PBXGroup;
\t\t\tchildren = (
{app_root_children}
\t\t\t\t{VIEWS_GRP} /* Views */,
\t\t\t\t{VIEWMODELS_GRP} /* ViewModels */,
\t\t\t\t{ASSETS_REF} /* Assets.xcassets */,
\t\t\t);
\t\t\tpath = SwiftBuddy;
\t\t\tsourceTree = "<group>";
\t\t}};
\t\t{VIEWS_GRP} /* Views */ = {{
\t\t\tisa = PBXGroup;
\t\t\tchildren = (
{views_children}
\t\t\t);
\t\t\tpath = Views;
\t\t\tsourceTree = "<group>";
\t\t}};
\t\t{VIEWMODELS_GRP} /* ViewModels */ = {{
\t\t\tisa = PBXGroup;
\t\t\tchildren = (
{viewmodel_children}
\t\t\t);
\t\t\tpath = ViewModels;
\t\t\tsourceTree = "<group>";
\t\t}};
/* End PBXGroup section */

/* Begin PBXNativeTarget section */
\t\t{APP_TARGET} /* SwiftBuddy */ = {{
\t\t\tisa = PBXNativeTarget;
\t\t\tbuildConfigurationList = {TGT_CFGLIST};
\t\t\tbuildPhases = (
\t\t\t\t{PHASE_SRC} /* Sources */,
\t\t\t\t{PHASE_FWK} /* Frameworks */,
\t\t\t\t{PHASE_RES} /* Resources */,
\t\t\t);
\t\t\tbuildRules = ();
\t\t\tdependencies = ();
\t\t\tname = SwiftBuddy;
\t\t\tpackageProductDependencies = (
\t\t\t\t{PROD_MLX} /* MLX */,
\t\t\t\t{PROD_MLXLLM} /* MLXLLM */,
\t\t\t\t{PROD_MLXLMC} /* MLXLMCommon */,
				{PROD_MLXVLM} /* MLXVLM */,
\t\t\t\t{PROD_HUB} /* Hub */,
\t\t\t\t{PROD_TOKENIZERS} /* Tokenizers */,
\t\t\t\t{PROD_HUMMINGBIRD} /* Hummingbird */,
\t\t\t\t{PROD_SWIFTSOUP} /* SwiftSoup */,
\t\t\t);
\t\t\tproductName = SwiftBuddy;
\t\t\tproductReference = {APP_PRODUCT};
\t\t\tproductType = "com.apple.product-type.application";
\t\t}};
/* End PBXNativeTarget section */

/* Begin PBXProject section */
\t\t{PROJ} /* Project object */ = {{
\t\t\tisa = PBXProject;
\t\t\tattributes = {{
\t\t\t\tBuildIndependentTargetsInParallel = 1;
\t\t\t\tLastSwiftUpdateCheck = 1540;
\t\t\t\tLastUpgradeCheck = 1540;
\t\t\t\tTargetAttributes = {{
\t\t\t\t\t{APP_TARGET} = {{
\t\t\t\t\t\tCreatedOnToolsVersion = 15.4;
\t\t\t\t\t}};
\t\t\t\t}};
\t\t\t}};
\t\t\tbuildConfigurationList = {PROJ_CFGLIST};
\t\t\tcompatibilityVersion = "Xcode 14.0";
\t\t\tdevelopmentRegion = en;
\t\t\thasScannedForEncodings = 0;
\t\t\tknownRegions = (en, Base);
\t\t\tmainGroup = {MAIN_GRP};
\t\t\tpackageReferences = (
\t\t\t\t{PKG_MLX} /* XCRemoteSwiftPackageReference "mlx-swift" */,
\t\t\t\t{PKG_MLXLM} /* XCRemoteSwiftPackageReference "mlx-swift-lm" */,
\t\t\t\t{PKG_TRANSFORMERS} /* XCRemoteSwiftPackageReference "swift-transformers" */,
\t\t\t\t{PKG_HUMMINGBIRD} /* XCRemoteSwiftPackageReference "hummingbird" */,
\t\t\t\t{PKG_SWIFTSOUP} /* XCRemoteSwiftPackageReference "SwiftSoup" */,
\t\t\t);
\t\t\tproductsGroup = {PRODUCTS_GRP};
\t\t\tprojectDirPath = "";
\t\t\tprojectRoot = "";
\t\t\ttargets = ({APP_TARGET});
\t\t}};
/* End PBXProject section */

/* Begin PBXResourcesBuildPhase section */
\t\t{PHASE_RES} /* Resources */ = {{
\t\t\tisa = PBXResourcesBuildPhase;
\t\t\tbuildActionMask = 2147483647;
\t\t\tfiles = ({ASSETS_BF} /* Assets.xcassets in Resources */);
\t\t\trunOnlyForDeploymentPostprocessing = 0;
\t\t}};
/* End PBXResourcesBuildPhase section */

/* Begin PBXSourcesBuildPhase section */
\t\t{PHASE_SRC} /* Sources */ = {{
\t\t\tisa = PBXSourcesBuildPhase;
\t\t\tbuildActionMask = 2147483647;
\t\t\tfiles = (
{src_build_files}
\t\t\t);
\t\t\trunOnlyForDeploymentPostprocessing = 0;
\t\t}};
/* End PBXSourcesBuildPhase section */

/* Begin XCBuildConfiguration section */
\t\t{PROJ_DEBUG} /* Debug */ = {{
\t\t\tisa = XCBuildConfiguration;
\t\t\tbuildSettings = {{
\t\t\t\tALWAYS_SEARCH_USER_PATHS = NO;
\t\t\t\tCOPY_PHASE_STRIP = NO;
\t\t\t\tDEBUG_INFORMATION_FORMAT = dwarf;
\t\t\t\tENABLE_TESTABILITY = YES;
\t\t\t\tGCC_OPTIMIZATION_LEVEL = 0;
\t\t\t\tMTL_ENABLE_DEBUG_INFO = INCLUDE_SOURCE;
\t\t\t\tMTL_FAST_MATH = YES;
\t\t\t\tONLY_ACTIVE_ARCH = YES;
\t\t\t\tARCHS = arm64;
\t\t\t\tVALID_ARCHS = arm64;
\t\t\t\tSWIFT_ACTIVE_COMPILATION_CONDITIONS = DEBUG;
\t\t\t}};
\t\t\tname = Debug;
\t\t}};
\t\t{PROJ_RELEASE} /* Release */ = {{
\t\t\tisa = XCBuildConfiguration;
\t\t\tbuildSettings = {{
\t\t\t\tALWAYS_SEARCH_USER_PATHS = NO;
\t\t\t\tCOPY_PHASE_STRIP = NO;
\t\t\t\tDEBUG_INFORMATION_FORMAT = "dwarf-with-dsym";
\t\t\t\tMTL_FAST_MATH = YES;
\t\t\t\tONLY_ACTIVE_ARCH = YES;
\t\t\t\tARCHS = arm64;
\t\t\t\tVALID_ARCHS = arm64;
\t\t\t\tSWIFT_COMPILATION_MODE = wholemodule;
\t\t\t}};
\t\t\tname = Release;
\t\t}};
\t\t{TGT_DEBUG} /* Debug */ = {{
\t\t\tisa = XCBuildConfiguration;
\t\t\tbuildSettings = {{
\t\t\t\tASSTCATALOG_COMPILER_APPICON_NAME = AppIcon;
\t\t\t\tASSTCATALOG_COMPILER_GLOBAL_ACCENT_COLOR_NAME = AccentColor;
\t\t\t\tCODE_SIGN_STYLE = Automatic;
\t\t\t\tCURRENT_PROJECT_VERSION = 1;
\t\t\t\tGENERATE_INFOPLIST_FILE = YES;
\t\t\t\tINFOPLIST_KEY_CFBundleDisplayName = "SwiftBuddy Chat";
\t\t\t\tINFOPLIST_KEY_NSHumanReadableCopyright = "Copyright © 2026 SharpAI";
\t\t\t\tINFOPLIST_KEY_UIApplicationSceneManifest_Generation = YES;
\t\t\t\tINFOPLIST_KEY_UIApplicationSupportsIndirectInputEvents = YES;
\t\t\t\tINFOPLIST_KEY_UILaunchScreen_Generation = YES;
\t\t\t\tINFOPLIST_KEY_UISupportedInterfaceOrientations_iPad = "UIInterfaceOrientationPortrait UIInterfaceOrientationPortraitUpsideDown UIInterfaceOrientationLandscapeLeft UIInterfaceOrientationLandscapeRight";
\t\t\t\tINFOPLIST_KEY_UISupportedInterfaceOrientations_iPhone = "UIInterfaceOrientationPortrait UIInterfaceOrientationLandscapeLeft UIInterfaceOrientationLandscapeRight";
\t\t\t\tIPHONEOS_DEPLOYMENT_TARGET = 17.0;
\t\t\t\tMACOS_DEPLOYMENT_TARGET = 14.0;
\t\t\t\tMARKETING_VERSION = 1.0;
\t\t\t\tPRODUCT_BUNDLE_IDENTIFIER = com.sharpai.SwiftBuddy;
\t\t\t\tPRODUCT_NAME = "$(TARGET_NAME)";
\t\t\t\tSDKROOT = auto;
\t\t\t\tSUPPORTED_PLATFORMS = "iphoneos iphonesimulator macosx";
\t\t\t\tSWIFT_EMIT_LOC_STRINGS = YES;
\t\t\t\tSWIFT_VERSION = 5.9;
\t\t\t\tTARGETED_DEVICE_FAMILY = "1,2";
\t\t\t}};
\t\t\tname = Debug;
\t\t}};
\t\t{TGT_RELEASE} /* Release */ = {{
\t\t\tisa = XCBuildConfiguration;
\t\t\tbuildSettings = {{
\t\t\t\tASSTCATALOG_COMPILER_APPICON_NAME = AppIcon;
\t\t\t\tASSTCATALOG_COMPILER_GLOBAL_ACCENT_COLOR_NAME = AccentColor;
\t\t\t\tCODE_SIGN_STYLE = Automatic;
\t\t\t\tCURRENT_PROJECT_VERSION = 1;
\t\t\t\tGENERATE_INFOPLIST_FILE = YES;
\t\t\t\tINFOPLIST_KEY_CFBundleDisplayName = "SwiftBuddy Chat";
\t\t\t\tIPHONEOS_DEPLOYMENT_TARGET = 17.0;
\t\t\t\tMACOS_DEPLOYMENT_TARGET = 14.0;
\t\t\t\tMARKETING_VERSION = 1.0;
\t\t\t\tPRODUCT_BUNDLE_IDENTIFIER = com.sharpai.SwiftBuddy;
\t\t\t\tPRODUCT_NAME = "$(TARGET_NAME)";
\t\t\t\tSDKROOT = auto;
\t\t\t\tSUPPORTED_PLATFORMS = "iphoneos iphonesimulator macosx";
\t\t\t\tSWIFT_EMIT_LOC_STRINGS = YES;
\t\t\t\tSWIFT_VERSION = 5.9;
\t\t\t\tTARGETED_DEVICE_FAMILY = "1,2";
\t\t\t}};
\t\t\tname = Release;
\t\t}};
/* End XCBuildConfiguration section */

/* Begin XCConfigurationList section */
\t\t{PROJ_CFGLIST} = {{
\t\t\tisa = XCConfigurationList;
\t\t\tbuildConfigurations = ({PROJ_DEBUG} /* Debug */, {PROJ_RELEASE} /* Release */);
\t\t\tdefaultConfigurationIsVisible = 0;
\t\t\tdefaultConfigurationName = Release;
\t\t}};
\t\t{TGT_CFGLIST} = {{
\t\t\tisa = XCConfigurationList;
\t\t\tbuildConfigurations = ({TGT_DEBUG} /* Debug */, {TGT_RELEASE} /* Release */);
\t\t\tdefaultConfigurationIsVisible = 0;
\t\t\tdefaultConfigurationName = Release;
\t\t}};
/* End XCConfigurationList section */

/* Begin XCLocalSwiftPackageReference section */
		{PKG_MLX} /* XCLocalSwiftPackageReference "mlx-swift" */ = {{
			isa = XCLocalSwiftPackageReference;
			relativePath = ../mlx-swift;
		}};
		{PKG_MLXLM} /* XCLocalSwiftPackageReference "mlx-swift-lm" */ = {{
			isa = XCLocalSwiftPackageReference;
			relativePath = ../mlx-swift-lm;
		}};
/* End XCLocalSwiftPackageReference section */

/* Begin XCRemoteSwiftPackageReference section */
\t\t{PKG_TRANSFORMERS} /* XCRemoteSwiftPackageReference "swift-transformers" */ = {{
\t\t\tisa = XCRemoteSwiftPackageReference;
\t\t\trepositoryURL = "https://github.com/huggingface/swift-transformers";
\t\t\trequirement = {{
\t\t\t\tkind = upToNextMinorVersion;
\t\t\t\tminimumVersion = 1.2.0;
\t\t\t}};
\t\t}};
\t\t{PKG_HUMMINGBIRD} /* XCRemoteSwiftPackageReference "hummingbird" */ = {{
\t\t\tisa = XCRemoteSwiftPackageReference;
\t\t\trepositoryURL = "https://github.com/hummingbird-project/hummingbird";
\t\t\trequirement = {{
\t\t\t\tkind = upToNextMajorVersion;
\t\t\t\tminimumVersion = 2.0.0;
\t\t\t}};
\t\t}};
\t\t{PKG_SWIFTSOUP} /* XCRemoteSwiftPackageReference "SwiftSoup" */ = {{
\t\t\tisa = XCRemoteSwiftPackageReference;
\t\t\trepositoryURL = "https://github.com/scinfu/SwiftSoup.git";
\t\t\trequirement = {{
\t\t\t\tkind = upToNextMajorVersion;
\t\t\t\tminimumVersion = 2.7.0;
\t\t\t}};
\t\t}};
/* End XCRemoteSwiftPackageReference section */

/* Begin XCSwiftPackageProductDependency section */
\t\t{PROD_MLX} /* MLX */ = {{
\t\t\tisa = XCSwiftPackageProductDependency;
\t\t\tpackage = {PKG_MLX};
\t\t\tproductName = MLX;
\t\t}};
\t\t{PROD_MLXLLM} /* MLXLLM */ = {{
\t\t\tisa = XCSwiftPackageProductDependency;
\t\t\tpackage = {PKG_MLXLM};
\t\t\tproductName = MLXLLM;
\t\t}};
\t\t{PROD_MLXLMC} /* MLXLMCommon */ = {{
\t\t\tisa = XCSwiftPackageProductDependency;
\t\t\tpackage = {PKG_MLXLM};
\t\t\tproductName = MLXLMCommon;
		}};
		{PROD_MLXVLM} /* MLXVLM */ = {{
			isa = XCSwiftPackageProductDependency;
			package = {PKG_MLXLM};
			productName = MLXVLM;
\t\t}};
\t\t{PROD_HUB} /* Hub */ = {{
\t\t\tisa = XCSwiftPackageProductDependency;
\t\t\tpackage = {PKG_TRANSFORMERS};
\t\t\tproductName = Hub;
\t\t}};
\t\t{PROD_TOKENIZERS} /* Tokenizers */ = {{
\t\t\tisa = XCSwiftPackageProductDependency;
\t\t\tpackage = {PKG_TRANSFORMERS};
\t\t\tproductName = Tokenizers;
\t\t}};
\t\t{PROD_HUMMINGBIRD} /* Hummingbird */ = {{
\t\t\tisa = XCSwiftPackageProductDependency;
\t\t\tpackage = {PKG_HUMMINGBIRD};
\t\t\tproductName = Hummingbird;
\t\t}};
\t\t{PROD_SWIFTSOUP} /* SwiftSoup */ = {{
\t\t\tisa = XCSwiftPackageProductDependency;
\t\t\tpackage = {PKG_SWIFTSOUP};
\t\t\tproductName = SwiftSoup;
\t\t}};
/* End XCSwiftPackageProductDependency section */

\t}};
\trootObject = {PROJ} /* Project object */;
}}
"""

def main():
    proj_dir = Path("SwiftBuddy.xcodeproj")
    proj_dir.mkdir(exist_ok=True)

    ws_dir = proj_dir / "project.xcworkspace"
    ws_dir.mkdir(exist_ok=True)
    (ws_dir / "contents.xcworkspacedata").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Workspace version = "1.0">\n'
        '   <FileRef location = "self:"></FileRef>\n'
        '</Workspace>\n'
    )

    pbx = proj_dir / "project.pbxproj"
    pbx.write_text(pbxproj())
    print("✅  SwiftBuddy.xcodeproj/project.pbxproj generated")
    print("✅  Workspace data written")
    print()
    print("📦  Local packages wired:")
    print("    • ../LocalPackages/mlx-swift  → MLX")
    print("    • ../mlx-swift-lm             → MLXLLM, MLXLMCommon")
    print()
    print("📂  MLXInferenceCore sources included directly:")
    for p, _, _ in [("ChatMessage", None, None), ("GenerationConfig", None, None),
                    ("ModelCatalog", None, None), ("ModelDownloadManager", None, None),
                    ("ModelArchitectureProbe", None, None), ("InferenceEngine", None, None)]:
        print(f"    • {p}.swift")
    print()
    print("🎉  Open SwiftBuddy.xcodeproj in Xcode — no manual package setup needed.")

if __name__ == "__main__":
    main()
