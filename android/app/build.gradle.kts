plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

dependencies {
    testImplementation("junit:junit:4.13.2")
}

android {
    namespace = "org.pokit.pokitlms"
    compileSdk = 35
    defaultConfig {
        applicationId = "org.pokit.pokitlms"
        minSdk = 24
        targetSdk = 28
        versionCode = 1
        versionName = "0.1.0"
        ndkVersion = "27.2.12479018"
        ndk { abiFilters += "arm64-v8a" }
        externalNativeBuild {
            cmake { arguments += listOf("-DPOKITLMS_BUILD_TESTS=OFF", "-DPOKITLMS_BUILD_BENCHMARK=OFF", "-DPOKITLMS_BUILD_ANDROID_APP=ON") }
        }
    }
    externalNativeBuild {
        cmake { path = file("src/main/cpp/CMakeLists.txt"); version = "3.22.1" }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
}
