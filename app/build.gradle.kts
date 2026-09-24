plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}
android {
    namespace="com.llmapp"
    compileSdk=35
    ndkVersion="27.2.12479018"
    defaultConfig {
        applicationId="com.llmapp"
        minSdk=28
        targetSdk=35
        versionCode=1
        versionName="1.0.0"
        ndk { abiFilters += listOf("arm64-v8a") }
        externalNativeBuild { cmake { cppFlags += listOf("-O3","-ffast-math","-fno-math-errno","-fno-signed-zeros","-ffp-contract=fast") } }
    }
    buildTypes { release { isMinifyEnabled=true; proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"),"proguard-rules.pro") } }
    compileOptions {
        sourceCompatibility=JavaVersion.VERSION_17
        targetCompatibility=JavaVersion.VERSION_17
    }
    kotlin {
        jvmToolchain(17)
    }
    externalNativeBuild { cmake { path=file("src/main/cpp/CMakeLists.txt"); version="3.30.5" } }
    packaging { jniLibs.useLegacyPackaging=true }
}
dependencies {
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.activity:activity-compose:1.10.1")
    implementation("androidx.compose.ui:ui:1.7.8")
    implementation("androidx.compose.foundation:foundation:1.7.8")
    implementation("androidx.compose.material3:material3:1.3.1")
    implementation("androidx.compose.material:material-icons-extended:1.7.8")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.7")
}
