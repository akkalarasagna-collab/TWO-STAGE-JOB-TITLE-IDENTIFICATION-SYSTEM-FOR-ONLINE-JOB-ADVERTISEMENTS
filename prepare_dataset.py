# prepare_dataset.py — Clean job_title_des.csv → merged_training_data.csv
# Includes hand-written high-quality samples for the 4 weakest titles.

import pandas as pd
import re

# ── Extra high-quality hand-written samples ───────────────────────────────────
extra_samples = [
    # ── Backend Developer — 8 discriminative samples ───────────────────────
    {"title": "Backend Developer", "description": "Backend developer to build server-side REST APIs using Python Django or FastAPI. Work with PostgreSQL database, Redis caching, Celery task queues, and Docker deployment. No frontend work required."},
    {"title": "Backend Developer", "description": "Pure backend developer role. Build and maintain REST APIs in Node.js Express. Design MongoDB schemas, implement JWT authentication, write unit tests with Jest, and deploy to AWS Lambda."},
    {"title": "Backend Developer", "description": "Backend engineer to develop microservices in Java Spring Boot. REST API design, Hibernate ORM, MySQL database, RabbitMQ message queues, Docker, and Kubernetes deployment experience required."},
    {"title": "Backend Developer", "description": "Server-side backend developer with Python experience. Build REST APIs using Flask, work with PostgreSQL and Redis, implement OAuth2 authentication, write API documentation with Swagger."},
    {"title": "Backend Developer", "description": "Backend API developer using Go or Python. Design scalable REST and GraphQL APIs, PostgreSQL database optimization, gRPC microservices, Docker containers, and AWS EC2 deployment."},
    {"title": "Backend Developer", "description": "Backend developer for fintech startup. Node.js REST APIs, MongoDB database design, Stripe payment integration, JWT authentication, rate limiting, and server-side security implementation."},
    {"title": "Backend Developer", "description": "Java backend developer needed for enterprise application. Spring Boot REST APIs, Oracle database, Hibernate, Maven build, JUnit testing, and CI/CD pipeline with Jenkins required."},
    {"title": "Backend Developer", "description": "Backend Python developer to build data processing APIs. FastAPI framework, async programming, PostgreSQL with SQLAlchemy, Redis pub/sub, Celery workers, and AWS S3 integration."},
    # ── Flutter Developer — 8 samples ─────────────────────────────────────
    {"title": "Flutter Developer", "description": "Develop cross-platform mobile applications using Flutter and Dart for both iOS and Android. Implement state management using Provider, Riverpod or Bloc. Integrate REST APIs and Firebase. Build pixel-perfect responsive UIs with Flutter widgets."},
    {"title": "Flutter Developer", "description": "Flutter mobile developer needed with strong Dart programming skills. Experience with Flutter SDK, Firebase Firestore, Firebase Auth, push notifications, and Google Maps integration required."},
    {"title": "Flutter Developer", "description": "Build and maintain Flutter applications for iOS and Android platforms. Work with Dart language, implement clean architecture, integrate third-party packages, and publish apps to Play Store and App Store."},
    {"title": "Flutter Developer", "description": "Seeking Flutter engineer with 2+ years Dart experience. Must know Flutter widget tree, animations, state management with GetX or Bloc, SQLite local storage, and RESTful API integration."},
    {"title": "Flutter Developer", "description": "Cross-platform mobile developer using Flutter framework. Responsibilities include UI development in Dart, integrating payment gateways, push notifications via FCM, and unit testing Flutter widgets."},
    {"title": "Flutter Developer", "description": "Flutter developer to create high-performance iOS and Android apps. Strong knowledge of Dart, Flutter SDK, Provider state management, Dio HTTP client, and Firebase backend services required."},
    {"title": "Flutter Developer", "description": "Mobile app developer with Flutter experience to build e-commerce application. Skills needed: Dart, Flutter, REST API, Firebase, BLoC pattern, Android Studio, and Xcode familiarity."},
    {"title": "Flutter Developer", "description": "Looking for skilled Flutter developer familiar with null safety Dart, Flutter 3.x, custom animations, platform channels for native iOS and Android code, and CI/CD with Fastlane."},
    # ── Wordpress Developer — 8 samples ────────────────────────────────────
    {"title": "Wordpress Developer", "description": "WordPress developer needed to build and maintain WordPress websites. Experience with PHP, custom theme development, plugin development, WooCommerce, and MySQL required."},
    {"title": "Wordpress Developer", "description": "Develop custom WordPress themes and plugins using PHP, HTML, CSS, and JavaScript. Experience with WooCommerce, Elementor, ACF Advanced Custom Fields, and WordPress REST API required."},
    {"title": "Wordpress Developer", "description": "WordPress developer to manage and update existing WordPress sites. Skills required: PHP, WordPress hooks and filters, custom post types, WooCommerce configuration, and website performance optimization."},
    {"title": "Wordpress Developer", "description": "Build WordPress websites from scratch using Elementor or Divi builder. Strong PHP and MySQL skills, experience with WordPress multisite, caching plugins, and site speed optimization needed."},
    {"title": "Wordpress Developer", "description": "Seeking WordPress and WooCommerce developer to build online stores. Must know PHP, MySQL, WordPress theme customization, WooCommerce extensions, payment gateway integration, and SEO plugins."},
    {"title": "Wordpress Developer", "description": "WordPress backend developer with PHP expertise to develop custom plugins, integrate third-party APIs, optimize database queries, and maintain WordPress security and updates."},
    {"title": "Wordpress Developer", "description": "Full WordPress developer role covering theme development with PHP and Blade templating, custom Gutenberg blocks in JavaScript, REST API endpoints, and WooCommerce store management."},
    {"title": "Wordpress Developer", "description": "WordPress CMS developer to create landing pages and marketing sites. Experience with Elementor, ACF, WPML multilingual plugin, WordPress SEO, caching with WP Rocket, and MySQL database."},
    # ── Software Engineer — 5 specific samples ──────────────────────────────
    {"title": "Software Engineer", "description": "Software engineer to design and build scalable backend systems. Strong knowledge of data structures, algorithms, system design, object-oriented programming. Experience with Python or Java and microservices architecture required."},
    {"title": "Software Engineer", "description": "Backend software engineer with experience in distributed systems, REST API design, SQL and NoSQL databases, cloud deployment on AWS or GCP, and automated testing with CI/CD pipelines."},
    {"title": "Software Engineer", "description": "Software engineer role focused on building enterprise software systems. Required skills: Java or C++, object-oriented design, design patterns, unit testing, code reviews, and Agile development methodology."},
    {"title": "Software Engineer", "description": "Mid-level software engineer to develop and maintain core platform services. Experience with Python, Go, or Java backend development, PostgreSQL, Redis caching, Docker containers, and Kubernetes deployment."},
    {"title": "Software Engineer", "description": "Software engineer for embedded systems and low-level programming. Experience with C, C++, Linux kernel, memory management, real-time operating systems, and hardware-software integration required."},
    # ── Full Stack Developer — 6 samples ────────────────────────────────────
    {"title": "Full Stack Developer", "description": "Full stack developer with React frontend and Node.js backend skills. Experience with MongoDB, REST API design, GraphQL, JWT authentication, Docker, and deploying to AWS or Heroku required."},
    {"title": "Full Stack Developer", "description": "Full stack web developer to build end-to-end web applications. Must know React or Vue.js for frontend, Node.js Express for backend, PostgreSQL or MongoDB database, and Git version control."},
    {"title": "Full Stack Developer", "description": "Seeking full stack engineer with Angular frontend and Spring Boot Java backend experience. AWS cloud deployment, MySQL database design, REST API development, and Docker containerization required."},
    {"title": "Full Stack Developer", "description": "Full stack JavaScript developer with MERN stack expertise. MongoDB, Express.js, React, Node.js. Experience with Redux state management, REST APIs, JWT tokens, and CI/CD deployment pipelines."},
    {"title": "Full Stack Developer", "description": "Full stack developer to build SaaS product. React TypeScript frontend, Python FastAPI backend, PostgreSQL database, Redis caching, Docker deployment, and AWS S3 storage integration."},
    {"title": "Full Stack Developer", "description": "Build and maintain full stack web application. Vue.js frontend with Vuex, Laravel PHP backend, MySQL database, REST API integration, Redis, and deployment on DigitalOcean or AWS."},
]

# ── Load and clean job_title_des.csv ─────────────────────────────────────────
df = pd.read_csv("job_title_des.csv")

# Drop duplicates
df = df.drop_duplicates(subset=["Job Description"])

# Rename columns
df = df.rename(columns={"Job Title": "title", "Job Description": "description"})

# Fix title names
df["title"] = df["title"].replace({
    "Machine Learning":  "Machine Learning Engineer",
    "Node js developer": "Node.js Developer",
})

# Drop the index column if present
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# ── Boilerplate cleaning ──────────────────────────────────────────────────────
BOILERPLATE_PATTERNS = [
    r"Job Types?:.*",
    r"Salary:.*",
    r"Pay:.*",
    r"Schedule:.*",
    r"Benefits:.*",
    r"Supplemental Pay:.*",
    r"Experience:.*",
    r"Work Remotely.*",
    r"Ability to commute.*",
    r"Housing.*subsidy.*",
    r"Food allowance.*",
    r"Joining bonus.*",
    r"Overtime pay.*",
    r"Industry:.*",
    r"Education:.*",
    r"Location:.*",
]


def clean_description(text):
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        skip = False
        for pattern in BOILERPLATE_PATTERNS:
            if re.match(pattern, line.strip(), re.IGNORECASE):
                skip = True
                break
        if not skip and len(line.strip()) > 2:
            cleaned.append(line.strip())
    result = " ".join(cleaned)
    if len(result) < 50:
        return text
    return result


df["description"] = df["description"].apply(clean_description)

# Drop descriptions shorter than 100 chars after cleaning
df = df[df["description"].str.len() >= 100]

# ── Append extra high-quality samples ────────────────────────────────────────
extra_df = pd.DataFrame(extra_samples)
df = pd.concat([df, extra_df], ignore_index=True)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Print summary
print(f"Total rows: {len(df)}")
print(f"\nRows per title:")
print(df["title"].value_counts())

# Save — write to a fresh temp file to avoid PermissionError if any CSV is open in Excel
import os

for TEMP_PATH in ["merged_train_temp.csv", "merged_training_data_new.csv", "merged_training_data.csv"]:
    try:
        df[["title", "description"]].to_csv(TEMP_PATH, index=False)
        print(f"\nSaved to {TEMP_PATH}")
        break
    except PermissionError:
        print(f"[warn] {TEMP_PATH} is locked, trying next filename...")
