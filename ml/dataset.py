# ml/dataset.py — Generate synthetic job description dataset and save to data/job_data.csv

import os
import csv
import random

random.seed(42)

# ---------------------------------------------------------------------------
# Raw sentence pools per job title
# Each pool has enough varied sentences to build 25+ unique descriptions
# ---------------------------------------------------------------------------

POOLS = {

"Software Engineer": [
    "Design, develop, and maintain high-quality software applications using modern programming languages.",
    "Collaborate with cross-functional teams to define, design, and ship new features.",
    "Write clean, scalable, and well-documented code following best practices.",
    "Participate in code reviews to ensure code quality and share knowledge with the team.",
    "Troubleshoot, debug, and upgrade existing software systems.",
    "Work closely with product managers and designers to translate requirements into technical solutions.",
    "Develop and maintain APIs and back-end services for web and mobile applications.",
    "Integrate third-party services and APIs into existing systems.",
    "Write unit tests and integration tests to ensure software reliability.",
    "Optimize application performance and improve system scalability.",
    "Contribute to architectural decisions and technical roadmap discussions.",
    "Mentor junior developers and provide technical guidance.",
    "Use version control systems such as Git to manage source code.",
    "Deploy applications to cloud platforms including AWS, Azure, or GCP.",
    "Follow Agile/Scrum methodologies and participate in sprint planning and retrospectives.",
    "Analyze user requirements and convert them into technical specifications.",
    "Research and adopt new technologies to improve development processes.",
    "Implement security best practices throughout the software development lifecycle.",
    "Work with databases including PostgreSQL, MySQL, and MongoDB.",
    "Continuously improve software delivery pipelines using CI/CD tools.",
    "Strong proficiency in Python, Java, or C++ is required for this role.",
    "Experience with microservices architecture and RESTful API design is expected.",
    "Familiarity with Docker and Kubernetes is a strong advantage.",
    "Excellent problem-solving skills and attention to detail are essential.",
    "A degree in Computer Science, Software Engineering, or a related field is preferred.",
    # ── 5 extra discriminative samples (varied stacks, no web/UI focus) ──
    "Develop low-level system utilities and libraries in C++ and Rust for embedded platforms.",
    "Own the full software development lifecycle from architecture through deployment for core platform services.",
    "Implement algorithms and data structures to solve computationally intensive engineering problems.",
    "Build internal developer tooling and automation frameworks used by the wider engineering organisation.",
    "Apply object-oriented and functional programming paradigms to deliver production-grade Java services.",
],

"Data Scientist": [
    "Analyze large and complex datasets to extract actionable business insights.",
    "Build and deploy predictive models using machine learning and statistical techniques.",
    "Collaborate with engineering and product teams to integrate models into production systems.",
    "Use Python and R to develop data analysis pipelines and experiments.",
    "Apply techniques such as regression, classification, clustering, and time-series forecasting.",
    "Communicate findings clearly to both technical and non-technical stakeholders.",
    "Design and execute A/B tests to evaluate product changes and business strategies.",
    "Clean, transform, and validate data from multiple structured and unstructured sources.",
    "Develop data visualizations using Matplotlib, Seaborn, Tableau, or Power BI.",
    "Work with big data platforms and SQL/NoSQL databases to query large datasets.",
    "Implement NLP pipelines for text classification and sentiment analysis tasks.",
    "Maintain and improve existing machine learning models in production.",
    "Stay current with the latest research in machine learning and data science.",
    "Document methodologies, experiments, and results for reproducibility.",
    "Strong knowledge of statistics, probability, and linear algebra is required.",
    "Experience with TensorFlow, PyTorch, or Scikit-learn is highly desirable.",
    "Familiarity with cloud platforms such as AWS SageMaker or GCP AI Platform is a plus.",
    "Work cross-functionally with business analysts and data engineers.",
    "Drive data-driven decision-making across the organization.",
    "Experience with Pandas, NumPy, and Jupyter Notebooks is expected.",
    "Ability to translate complex analytical findings into business recommendations.",
    "Proficiency in SQL for data extraction and exploration is required.",
    "Knowledge of feature engineering and model selection techniques is essential.",
    "Experience deploying machine learning models using REST APIs is advantageous.",
    "A Master's or PhD in Statistics, Mathematics, Computer Science, or a related field is preferred.",
],

"Machine Learning Engineer": [
    "Design, build, and deploy scalable machine learning models and pipelines.",
    "Collaborate with data scientists to bring ML prototypes into production-grade systems.",
    "Develop and maintain training, evaluation, and inference pipelines.",
    "Optimize ML models for latency, throughput, and resource efficiency.",
    "Work with large-scale datasets stored in distributed systems.",
    "Implement feature engineering, model versioning, and experiment tracking.",
    "Use frameworks such as TensorFlow, PyTorch, and Scikit-learn to build models.",
    "Build and maintain MLOps infrastructure including model monitoring and retraining pipelines.",
    "Integrate ML models into production applications via REST APIs.",
    "Write efficient Python code and ensure models meet performance benchmarks.",
    "Partner with DevOps teams to containerize and deploy models using Docker and Kubernetes.",
    "Develop deep learning architectures for classification, regression, and generation tasks.",
    "Monitor deployed models for performance drift and data quality issues.",
    "Contribute to the selection and evaluation of ML frameworks and tools.",
    "Strong understanding of supervised and unsupervised learning algorithms is required.",
    "Experience with cloud-based ML services such as AWS SageMaker or Azure ML is preferred.",
    "Proficiency in Python and familiarity with C++ for performance-critical components.",
    "Knowledge of MLflow, Kubeflow, or similar ML lifecycle management tools is a plus.",
    "Solid background in linear algebra, calculus, and probability theory is expected.",
    "Experience with data pipelines using Apache Spark or Apache Kafka is advantageous.",
    "Ability to communicate technical concepts clearly to both engineers and business stakeholders.",
    "Work in an Agile environment with regular sprint cycles and code reviews.",
    "Familiarity with A/B testing and experimentation frameworks is desired.",
    "Experience with transformer architectures and pre-trained language models is a strong plus.",
    "A Bachelor's or Master's degree in Computer Science, Mathematics, or a related field is required.",
],

"DevOps Engineer": [
    "Design, implement, and maintain CI/CD pipelines to automate software delivery.",
    "Manage and provision infrastructure using Infrastructure-as-Code tools such as Terraform or Ansible.",
    "Monitor system health, availability, and performance using tools like Prometheus and Grafana.",
    "Work closely with development teams to improve deployment frequency and reliability.",
    "Containerize applications and manage Kubernetes clusters in production environments.",
    "Administer Linux servers and ensure system security and stability.",
    "Automate repetitive operational tasks using Bash, Python, or PowerShell scripts.",
    "Configure and maintain cloud environments on AWS, Azure, or GCP.",
    "Implement logging, alerting, and observability solutions across the infrastructure.",
    "Collaborate on disaster recovery planning and business continuity strategies.",
    "Manage source code repositories and branching strategies with Git and GitHub.",
    "Support development teams in adopting DevOps best practices and tooling.",
    "Perform capacity planning and optimize cloud resource utilization.",
    "Ensure secure configuration of infrastructure components and access controls.",
    "Experience with Jenkins, GitLab CI, or GitHub Actions is required.",
    "Proficiency with Docker and Kubernetes is essential for this role.",
    "Knowledge of networking concepts including DNS, load balancing, and firewalls is expected.",
    "Hands-on experience with AWS services such as EC2, S3, RDS, and Lambda is preferred.",
    "Familiarity with configuration management tools like Chef, Puppet, or Ansible is a plus.",
    "Strong scripting skills in Bash or Python are required.",
    "Experience with service mesh technologies such as Istio is advantageous.",
    "Ability to respond to and resolve production incidents effectively.",
    "Understanding of Agile and Scrum delivery methodologies.",
    "Strong communication and collaboration skills are essential.",
    "A degree in Computer Science, Information Technology, or a related field is preferred.",
],

"Business Analyst": [
    "Gather, analyze, and document business requirements from stakeholders.",
    "Bridge the gap between business units and IT development teams.",
    "Create detailed business requirement documents, user stories, and process flows.",
    "Conduct gap analysis to identify areas of improvement in existing processes.",
    "Facilitate workshops, interviews, and focus groups with key stakeholders.",
    "Analyze business processes and recommend solutions to improve efficiency.",
    "Develop and maintain business intelligence dashboards and reports.",
    "Work with project managers to ensure project deliverables align with business goals.",
    "Support UAT (User Acceptance Testing) and validate that solutions meet requirements.",
    "Create data-driven presentations to communicate insights to executive stakeholders.",
    "Collaborate with technical teams to translate business needs into functional specifications.",
    "Use tools such as JIRA, Confluence, and Visio for documentation and tracking.",
    "Identify risks and dependencies and escalate issues proactively.",
    "Monitor and evaluate the performance of implemented solutions.",
    "Proficiency in Excel, Tableau, or Power BI for data analysis and reporting is required.",
    "Strong understanding of business process modeling using BPMN is preferred.",
    "Experience with Agile methodologies and writing user stories is highly desirable.",
    "Excellent written and verbal communication skills are essential.",
    "Ability to manage multiple projects simultaneously and meet deadlines.",
    "Knowledge of SQL for basic data querying and validation is an advantage.",
    "Experience in the finance, healthcare, or retail sector is beneficial.",
    "Strong analytical thinking and problem-solving capabilities.",
    "Familiarity with ERP systems such as SAP or Oracle is a plus.",
    "A degree in Business Administration, Information Systems, or a related field is required.",
    "Attention to detail and strong organizational skills are expected.",
],

"Full Stack Developer": [
    "Develop and maintain both front-end and back-end components of web applications.",
    "Build responsive user interfaces using React, Angular, or Vue.js.",
    "Develop RESTful APIs and server-side logic using Node.js, Django, or Flask.",
    "Work with relational and non-relational databases such as PostgreSQL and MongoDB.",
    "Collaborate with UI/UX designers to implement pixel-perfect designs.",
    "Integrate third-party services, payment gateways, and external APIs.",
    "Write clean, maintainable code with proper documentation and unit tests.",
    "Optimize web applications for maximum speed and scalability.",
    "Use version control workflows with Git and follow branching strategies.",
    "Deploy applications to cloud platforms and configure hosting environments.",
    "Participate in Agile sprints, code reviews, and daily stand-ups.",
    "Troubleshoot cross-browser compatibility issues and resolve performance bottlenecks.",
    "Implement authentication and authorization mechanisms such as OAuth and JWT.",
    "Ensure web accessibility standards are met across all user interfaces.",
    "Experience with both JavaScript and TypeScript is required.",
    "Proficiency in HTML5, CSS3, and modern front-end tooling (Webpack, Vite) is expected.",
    "Familiarity with containerization using Docker is advantageous.",
    "Knowledge of CI/CD practices and automated testing is a plus.",
    "Strong understanding of HTTP protocols, REST APIs, and web security.",
    "Ability to work independently and manage tasks with minimal supervision.",
    "Experience with state management libraries such as Redux or Vuex.",
    "Understanding of microservices architecture is beneficial.",
    "Good communication skills and ability to collaborate with remote teams.",
    "A degree in Computer Science or Software Engineering is preferred.",
    "Passion for building user-friendly, scalable web products.",
    # ── 5 extra discriminative samples (explicit full-stack ownership) ──
    "Own both the React front-end and the Node.js/Express back-end of a customer-facing SaaS product.",
    "Span the entire stack: craft Tailwind CSS components in the morning and optimise PostgreSQL queries in the afternoon.",
    "Architect end-to-end features across Vue.js, a GraphQL gateway, and a MongoDB data layer.",
    "Switch fluidly between writing TypeScript React components and building Django REST framework endpoints.",
    "Deliver features independently across the client, server, and database tiers of a Next.js e-commerce platform.",
],

"Data Analyst": [
    "Collect, clean, and analyze data to support business decision-making processes.",
    "Create dashboards, reports, and visualizations using Tableau, Power BI, or Excel.",
    "Write complex SQL queries to extract and transform data from relational databases.",
    "Collaborate with stakeholders to understand data requirements and deliver insights.",
    "Identify trends, anomalies, and patterns in large datasets.",
    "Maintain and update existing reports and analytical models.",
    "Support data governance and data quality initiatives within the organization.",
    "Perform ad-hoc analysis to answer business questions quickly and accurately.",
    "Present findings and recommendations to non-technical audiences clearly.",
    "Work with data engineers to ensure data pipelines deliver accurate data.",
    "Develop KPIs and metrics to track business performance.",
    "Automate repetitive reporting tasks using Python or R scripts.",
    "Validate data accuracy by cross-referencing multiple data sources.",
    "Assist in the design of data collection systems and databases.",
    "Strong proficiency in SQL and Excel is required.",
    "Experience with Tableau or Power BI is highly desirable.",
    "Familiarity with Python (Pandas, NumPy) for data manipulation is a plus.",
    "Knowledge of statistical analysis methods is expected.",
    "Excellent communication and data storytelling skills are essential.",
    "Attention to detail and ability to work with large, complex datasets.",
    "Experience with Google Analytics or similar web analytics tools is beneficial.",
    "Ability to manage multiple analysis requests with varying priorities.",
    "Understanding of data warehousing concepts and ETL processes is a plus.",
    "A degree in Statistics, Mathematics, Computer Science, or a related field is preferred.",
    "Strong organizational skills and ability to document analytical processes thoroughly.",
],

"AI Engineer": [
    "Design and implement artificial intelligence solutions for real-world business problems.",
    "Build, train, and evaluate deep learning models for classification, detection, and generation tasks.",
    "Work with large language models (LLMs) and fine-tune pre-trained models for specific domains.",
    "Develop AI-powered APIs and microservices for integration into production systems.",
    "Collaborate with research teams to translate cutting-edge AI research into deployable products.",
    "Implement reinforcement learning and generative AI pipelines.",
    "Optimize model inference pipelines for production-scale deployment.",
    "Maintain up-to-date knowledge of AI research, frameworks, and tooling.",
    "Use cloud AI platforms including AWS Bedrock, Azure OpenAI, or GCP Vertex AI.",
    "Develop evaluation benchmarks and testing frameworks for AI system quality assurance.",
    "Work with transformer architectures including BERT, GPT, and T5.",
    "Implement RAG (Retrieval-Augmented Generation) pipelines for enterprise knowledge systems.",
    "Integrate AI models into web and mobile applications via REST APIs.",
    "Ensure AI systems are fair, explainable, and compliant with ethical guidelines.",
    "Proficiency in Python and deep learning frameworks such as PyTorch and TensorFlow is required.",
    "Experience with Hugging Face Transformers and model fine-tuning is highly desirable.",
    "Strong understanding of neural network architectures and optimization techniques.",
    "Familiarity with vector databases such as Pinecone, Weaviate, or FAISS is a plus.",
    "Knowledge of MLOps practices including experiment tracking and model monitoring.",
    "Experience deploying models at scale using Docker and Kubernetes.",
    "Strong mathematical background in linear algebra, calculus, and probability.",
    "Ability to communicate AI concepts clearly to non-technical stakeholders.",
    "A Master's or PhD in Artificial Intelligence, Computer Science, or a related field is preferred.",
    "Curiosity and passion for building innovative AI-driven products.",
    "Experience with autonomous agents and AI orchestration frameworks is advantageous.",
],

"Cloud Engineer": [
    "Design, implement, and manage cloud infrastructure on AWS, Azure, or GCP.",
    "Automate infrastructure provisioning using Terraform, CloudFormation, or Pulumi.",
    "Implement cloud security best practices including IAM policies, encryption, and network segmentation.",
    "Optimize cloud resource usage to reduce costs while maintaining performance.",
    "Set up and manage virtual networks, load balancers, and VPNs in cloud environments.",
    "Monitor cloud infrastructure health and respond to incidents and alerts.",
    "Migrate on-premises workloads to cloud environments with minimal disruption.",
    "Collaborate with development teams to architect cloud-native applications.",
    "Implement disaster recovery and backup strategies in the cloud.",
    "Maintain documentation for cloud architecture and operational procedures.",
    "Configure and manage serverless services such as AWS Lambda or Azure Functions.",
    "Work with managed database services including RDS, DynamoDB, and Cloud SQL.",
    "Build and maintain CI/CD pipelines integrated with cloud deployment workflows.",
    "Evaluate and recommend cloud services and architectural patterns.",
    "Hands-on experience with AWS, Azure, or GCP is required.",
    "Proficiency with Terraform or similar IaC tools is essential.",
    "Knowledge of cloud networking and security architecture is expected.",
    "Experience with containerization using Docker and orchestration with Kubernetes.",
    "Relevant cloud certifications (AWS Solutions Architect, Azure Administrator) are preferred.",
    "Strong scripting skills in Python or Bash for automation tasks.",
    "Familiarity with monitoring tools such as CloudWatch, Azure Monitor, or Stackdriver.",
    "Understanding of FinOps principles and cloud cost management.",
    "Strong problem-solving skills and ability to work under pressure during incidents.",
    "A degree in Computer Science, IT, or a related field is preferred.",
    "Excellent teamwork and communication skills are required.",
],

"Cyber Security Analyst": [
    "Monitor and analyze security events from SIEM platforms to detect threats and anomalies.",
    "Conduct vulnerability assessments and penetration testing on systems and applications.",
    "Respond to security incidents, perform root cause analysis, and implement remediation.",
    "Develop and enforce security policies, standards, and procedures.",
    "Perform threat intelligence analysis to stay ahead of emerging attack vectors.",
    "Administer and configure security tools including firewalls, IDS/IPS, and EDR solutions.",
    "Conduct security awareness training for employees across the organization.",
    "Review and audit access controls, user privileges, and identity management systems.",
    "Work with development teams to integrate security into the SDLC (DevSecOps).",
    "Prepare security reports and communicate risks to executive stakeholders.",
    "Investigate phishing attacks, malware infections, and data breach attempts.",
    "Maintain compliance with security frameworks such as ISO 27001, NIST, or SOC 2.",
    "Participate in red team and blue team exercises to test organizational defenses.",
    "Manage and respond to bug bounty program submissions.",
    "Experience with SIEM tools such as Splunk, QRadar, or Microsoft Sentinel is required.",
    "Proficiency in network security concepts including TCP/IP, DNS, and VPN is essential.",
    "Knowledge of ethical hacking methodologies and tools (Metasploit, Burp Suite, Nmap).",
    "Familiarity with cloud security best practices and tools is a strong plus.",
    "Relevant certifications such as CISSP, CEH, or CompTIA Security+ are preferred.",
    "Strong analytical skills and ability to think like an attacker.",
    "Experience with scripting languages (Python, Bash) for security automation.",
    "Understanding of cryptography, PKI, and secure communications.",
    "A degree in Cyber Security, Computer Science, or Information Technology is required.",
    "Ability to work under pressure and respond to incidents outside business hours.",
    "Strong attention to detail and commitment to maintaining confidentiality.",
],

"Frontend Developer": [
    "Build and maintain responsive, high-performance user interfaces for web applications.",
    "Translate UI/UX designs into clean, semantic HTML, CSS, and JavaScript code.",
    "Develop reusable components using React, Angular, or Vue.js.",
    "Optimize front-end performance by minimizing load times and improving rendering efficiency.",
    "Collaborate closely with back-end developers to integrate APIs and data services.",
    "Ensure cross-browser and cross-device compatibility of all UI components.",
    "Implement state management solutions using Redux, MobX, or Vuex.",
    "Write unit tests and end-to-end tests for front-end components.",
    "Use CSS preprocessors such as SASS or LESS and utility frameworks like Tailwind CSS.",
    "Work within Agile teams and contribute to sprint planning and backlog grooming.",
    "Participate in UI code reviews and maintain front-end coding standards.",
    "Implement accessibility (WCAG) standards to ensure inclusive user experiences.",
    "Use build tools such as Webpack, Vite, or Parcel for asset bundling.",
    "Integrate analytics and monitoring tools into web applications.",
    "Strong proficiency in HTML5, CSS3, and modern JavaScript (ES6+) is required.",
    "Experience with React.js or Vue.js is essential for this position.",
    "Familiarity with TypeScript is highly desirable.",
    "Knowledge of RESTful APIs and experience consuming them from the browser.",
    "Understanding of browser developer tools and performance profiling.",
    "Experience with responsive design and mobile-first development approaches.",
    "Ability to work collaboratively with UI/UX designers using Figma or Sketch.",
    "Familiarity with CI/CD workflows and version control with Git.",
    "Strong eye for detail and passion for building great user experiences.",
    "Good communication skills and ability to work in a distributed team.",
    "A degree in Computer Science, Web Development, or a related discipline is preferred.",
    # ── 5 extra discriminative samples (UI-only, no server-side ownership) ──
    "Focus exclusively on the browser layer: pixel-perfect HTML, CSS animations, and JavaScript interactions.",
    "Craft performant React component libraries consumed by multiple product teams, with no server-side responsibilities.",
    "Specialise in Core Web Vitals optimisation, Lighthouse audits, and lazy-loading strategies for a high-traffic portal.",
    "Own the design-system implementation in Storybook, converting Figma tokens into reusable Tailwind components.",
    "Drive progressive web app (PWA) features including service workers, offline caching, and push notifications.",
],

"Backend Developer": [
    "Design and develop robust, scalable back-end services and APIs.",
    "Build server-side logic using Python (Django/Flask/FastAPI), Node.js, or Java Spring Boot.",
    "Design and manage relational and non-relational databases including PostgreSQL and MongoDB.",
    "Implement authentication, authorization, and security controls in back-end systems.",
    "Integrate with third-party APIs, messaging queues, and external data sources.",
    "Write unit tests, integration tests, and ensure code quality through reviews.",
    "Optimize database queries and back-end performance for high-traffic applications.",
    "Document APIs using tools such as Swagger or Postman.",
    "Deploy and monitor back-end services in cloud environments.",
    "Collaborate with front-end developers to define API contracts and data models.",
    "Implement caching strategies using Redis or Memcached.",
    "Develop event-driven architectures using message brokers like Kafka or RabbitMQ.",
    "Ensure back-end services meet availability, scalability, and maintainability requirements.",
    "Participate in architectural design discussions and contribute to technical decisions.",
    "Proficiency in Python, Java, Node.js, or Go is required.",
    "Strong SQL skills and database design experience are essential.",
    "Experience with RESTful and GraphQL API design is expected.",
    "Familiarity with Docker and cloud deployment platforms is advantageous.",
    "Knowledge of microservices and distributed systems architecture is preferred.",
    "Understanding of software design patterns such as MVC, CQRS, and event sourcing.",
    "Experience with CI/CD pipelines and automated testing frameworks.",
    "Good understanding of data security and GDPR compliance requirements.",
    "Excellent problem-solving skills and ability to debug complex back-end issues.",
    "A degree in Computer Science, Software Engineering, or equivalent practical experience.",
    "Strong collaboration and communication skills are valued.",
    # ── 5 extra discriminative samples (pure server-side, no UI) ──
    "Build high-throughput gRPC microservices in Go with no involvement in front-end or UI work.",
    "Own the server-side domain: schema design, stored procedures, and query optimisation in PostgreSQL.",
    "Develop and expose FastAPI endpoints consumed by mobile and web clients; front-end is handled by a separate team.",
    "Implement background job processing with Celery and Redis to handle asynchronous workloads at scale.",
    "Maintain Spring Boot services that power the core business logic layer, leaving all UI concerns to frontend engineers.",
],

"Product Manager": [
    "Define and communicate the product vision, strategy, and roadmap.",
    "Gather and prioritize product requirements from customers, stakeholders, and market research.",
    "Work closely with engineering, design, and marketing teams to deliver product features.",
    "Write detailed product requirement documents and user stories for development teams.",
    "Conduct competitive analysis and market research to inform product decisions.",
    "Define and track key product metrics including activation, retention, and revenue.",
    "Lead product discovery processes including user interviews and usability testing.",
    "Manage the product backlog and prioritize features based on business value.",
    "Facilitate sprint planning, backlog grooming, and retrospective meetings.",
    "Communicate product updates and milestones to executive leadership and stakeholders.",
    "Identify and mitigate product risks throughout the development lifecycle.",
    "Collaborate with sales and customer success teams to ensure customer satisfaction.",
    "Make data-driven decisions by analyzing user behavior and A/B test results.",
    "Champion the voice of the customer throughout the organization.",
    "Experience with Agile product management and Scrum frameworks is required.",
    "Proficiency in tools such as JIRA, Confluence, Productboard, or Notion.",
    "Strong analytical skills and experience using data to drive product decisions.",
    "Excellent communication and presentation skills for diverse audiences.",
    "Ability to lead cross-functional teams without formal authority.",
    "Experience with B2B or B2C SaaS products is highly desirable.",
    "Knowledge of UX design principles and ability to work closely with designers.",
    "Technical background or experience working closely with engineering teams.",
    "Strong leadership, collaboration, and decision-making capabilities.",
    "A degree in Business, Computer Science, or a related field is preferred.",
    "Demonstrated track record of shipping successful products is required.",
],

}

# ---------------------------------------------------------------------------
# Build dataset: sample 25 descriptions per title → 325 samples total
# ---------------------------------------------------------------------------

def build_dataset(samples_per_title: int = 25):
    """
    Generate a list of (description, title) tuples by randomly combining
    5-8 sentences from each title's pool without repeating the same combination.
    """
    rows = []
    for title, sentences in POOLS.items():
        seen = set()
        attempts = 0
        while len(rows) < (list(POOLS.keys()).index(title) + 1) * samples_per_title:
            if attempts > samples_per_title * 10:
                break  # safety valve
            attempts += 1
            n = random.randint(5, 8)
            chosen = tuple(random.sample(sentences, min(n, len(sentences))))
            key = frozenset(chosen)
            if key in seen:
                continue
            seen.add(key)
            description = ' '.join(chosen)
            rows.append({'description': description, 'title': title})
    return rows


def save_csv(rows, filepath: str):
    """Write rows to a CSV file with columns: description, title."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['description', 'title'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[dataset.py] Saved {len(rows)} samples to {filepath}")


if __name__ == '__main__':
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'job_data.csv')
    output_path = os.path.normpath(output_path)

    dataset = build_dataset(samples_per_title=25)
    random.shuffle(dataset)
    save_csv(dataset, output_path)
    print(f"[dataset.py] Dataset generation complete. Total samples: {len(dataset)}")
