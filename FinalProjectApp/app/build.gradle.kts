import java.util.Properties

plugins {
    alias(libs.plugins.android.application)
}

// 서버 주소는 리포에 커밋하지 않는다. local.properties(gitignore됨)의
//   serverUrl=http://<내 서버 IP>:5000/
// 값을 빌드 시 BuildConfig.SERVER_URL로 주입한다. 없으면 운영 서버를 사용한다.
val serverUrl: String = run {
    val props = Properties()
    rootProject.file("local.properties").takeIf { it.exists() }?.inputStream()?.use { props.load(it) }
    props.getProperty("serverUrl")
        ?: (project.findProperty("serverUrl") as String?)
        ?: System.getenv("FINALINZI_SERVER_URL")
        ?: "https://finalinzi.onrender.com/"
}

android {
    namespace = "com.example.finalprojectapp"
    compileSdk = 37

    buildFeatures {
        buildConfig = true
    }

    defaultConfig {
        applicationId = "com.example.finalprojectapp"
        minSdk = 26
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        buildConfigField("String", "SERVER_URL", "\"$serverUrl\"")

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            optimization {
                enable = false
            }
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
}

dependencies {
    implementation(libs.androidx.activity.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.androidx.constraintlayout)
    implementation(libs.androidx.core.ktx)
    implementation(libs.material)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.androidx.junit)
}

tasks.matching { it.name == "preReleaseBuild" }.configureEach {
    doFirst {
        check(serverUrl.startsWith("https://")) {
            "Release serverUrl must use HTTPS"
        }
    }
}
