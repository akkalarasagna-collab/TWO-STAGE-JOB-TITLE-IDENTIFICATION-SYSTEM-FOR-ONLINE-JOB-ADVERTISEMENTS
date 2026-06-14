# app/skill_extractor.py — Extract technical and soft skills from job descriptions

import re


class SkillExtractor:
    """
    Scans a job description for known technical and soft skills using
    case-insensitive keyword matching with word-boundary checks.

    Returns two sorted lists: technical_skills, soft_skills.
    """

    # ── Technical skills vocabulary ──────────────────────────────────────────
    TECHNICAL_SKILLS = [
        # Languages
        'Python', 'Java', 'JavaScript', 'TypeScript', 'C++', 'C#', 'R', 'Bash',
        # Databases
        'SQL', 'NoSQL', 'MongoDB', 'PostgreSQL', 'MySQL',
        # Web / Frontend
        'HTML', 'CSS', 'React', 'Angular', 'Vue', 'Node.js',
        # Frameworks / Back-end
        'Django', 'Flask', 'FastAPI', 'Spring Boot',
        # Cloud & DevOps
        'AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'Jenkins', 'Git',
        'GitHub', 'Linux', 'CI/CD',
        # ML / AI
        'TensorFlow', 'PyTorch', 'Keras', 'Scikit-learn', 'Pandas', 'NumPy',
        'Matplotlib', 'Machine Learning', 'Deep Learning', 'NLP',
        'Computer Vision',
        # Analytics / BI
        'Data Analysis', 'Data Visualization', 'Tableau', 'Power BI', 'Excel',
        # Architecture / APIs
        'REST API', 'GraphQL', 'Microservices',
        # Process / Tools
        'Agile', 'Scrum', 'JIRA',
        # Design
        'Figma', 'Adobe XD',
    ]

    # ── Soft skills vocabulary ───────────────────────────────────────────────
    SOFT_SKILLS = [
        'Communication', 'Leadership', 'Teamwork', 'Problem Solving',
        'Critical Thinking', 'Time Management', 'Adaptability',
        'Attention to Detail', 'Collaboration', 'Creativity',
        'Decision Making', 'Analytical Thinking', 'Project Management',
        'Presentation Skills', 'Interpersonal Skills',
    ]

    def __init__(self):
        # Pre-compile patterns for each skill to avoid recompilation on every call.
        # Skills with special characters (C++, C#, Node.js, etc.) are escaped.
        self._tech_patterns = [
            (skill, re.compile(rf'\b{re.escape(skill)}\b', re.IGNORECASE))
            for skill in self.TECHNICAL_SKILLS
        ]
        self._soft_patterns = [
            (skill, re.compile(rf'\b{re.escape(skill)}\b', re.IGNORECASE))
            for skill in self.SOFT_SKILLS
        ]

    def extract(self, text: str):
        """
        Scan *text* and return matched skills.

        Returns:
            technical_skills (list[str]): Detected technical skills (canonical form).
            soft_skills      (list[str]): Detected soft skills (canonical form).
        """
        found_technical = [
            skill for skill, pattern in self._tech_patterns
            if pattern.search(text)
        ]
        found_soft = [
            skill for skill, pattern in self._soft_patterns
            if pattern.search(text)
        ]

        return found_technical, found_soft
