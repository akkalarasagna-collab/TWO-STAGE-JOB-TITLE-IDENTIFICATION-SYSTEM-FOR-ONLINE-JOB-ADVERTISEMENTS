# app/stage1_matcher.py — Stage 1: Keyword & Regex-based job title detection

import re


class Stage1Matcher:
    """
    Two-pass job title detector:
      Pass 1 — STRONG_PATTERNS: single-keyword signals so distinctive they
               identify a title unambiguously even without "developer/engineer"
               (e.g. "flutter", "dart sdk", "woocommerce", "laravel", "xcode").
               Returns the title with confidence 90 when any pattern fires.
      Pass 2 — REGEX_PATTERNS: explicit title phrases (e.g. "flutter developer",
               "devops engineer").  Returns title with confidence 100.
      Pass 3 — Exact KNOWN_TITLES keyword scan.

    Returns (title, confidence) tuple, or (None, None) if no match.
    """

    # ── Pass 1: Strong single-signal patterns ────────────────────────────────
    # Each entry: (compiled_regex, canonical_title)
    # Order matters — more specific patterns first.
    # Confidence returned: 90 (strong signal, not an explicit title mention)
    STRONG_PATTERNS = [
        # ── Flutter / Dart ──────────────────────────────────────────────────
        (r'\bflutter\s+(sdk|developer|engineer|framework|widget|app)\b',
                                                        'Flutter Developer'),
        (r'\bflutter\b',                                'Flutter Developer'),
        (r'\bdart\s+(sdk|programming|language|null\s+safety)\b',
                                                        'Flutter Developer'),
        (r'\bbloc\s+(pattern|state\s+management)\b.*\b(dart|flutter)\b',
                                                        'Flutter Developer'),
        (r'\b(dart|flutter)\b.*\bbloc\b',               'Flutter Developer'),
        (r'\briverpod\b',                               'Flutter Developer'),
        # ── WordPress / WooCommerce ─────────────────────────────────────────
        (r'\bwordpress\b',                              'Wordpress Developer'),
        (r'\bwoocommerce\b',                            'Wordpress Developer'),
        (r'\bwp\s+(developer|engineer|plugin|theme)\b', 'Wordpress Developer'),
        (r'\bgutenberg\b',                              'Wordpress Developer'),
        (r'\belementor\b',                              'Wordpress Developer'),
        (r'\badvanced\s+custom\s+fields\b',             'Wordpress Developer'),
        (r'\bacf\b.*\b(wordpress|wp|plugin)\b',         'Wordpress Developer'),
        # ── iOS / Swift / Xcode ─────────────────────────────────────────────
        (r'\bxcode\b',                                  'iOS Developer'),
        (r'\bswift\b.*\b(ios|apple|xcode|iphone|ipad)\b',
                                                        'iOS Developer'),
        (r'\b(ios|apple|xcode|iphone|ipad)\b.*\bswift\b',
                                                        'iOS Developer'),
        (r'\bobjective[-\s]?c\b',                       'iOS Developer'),
        (r'\bios\s+developer\b',                        'iOS Developer'),
        (r'\bapp\s+store\b.*\b(swift|ios|xcode)\b',     'iOS Developer'),
        # ── Django ──────────────────────────────────────────────────────────
        (r'\bdjango\s+(developer|rest\s+framework|orm|views|models)\b',
                                                        'Django Developer'),
        (r'\bdjango\b',                                 'Django Developer'),
        (r'\bdjango\s+rest\s+framework\b',              'Django Developer'),
        # ── Node.js / Express ───────────────────────────────────────────────
        # Require "node.js developer/engineer" phrasing, or node+express without
        # a frontend/full-stack signal that would make it Full Stack instead.
        # The negative lookahead is handled by checking in the match() logic;
        # here we use tighter compound patterns that need backend-exclusive context.
        (r'\bnode\.?js\s+(developer|engineer)\b',           'Node.js Developer'),
        (r'\bnode\.?js\b.*\bexpress\.?js\b.*\b(mongodb|mongoose|sequelize|postgresql)\b',
                                                            'Node.js Developer'),
        (r'\bexpress\.?js\b.*\bnode\.?js\b.*\b(mongodb|mongoose|sequelize|postgresql)\b',
                                                            'Node.js Developer'),
        # ── PHP / Laravel / CodeIgniter ─────────────────────────────────────
        (r'\blaravel\b',                                'PHP Developer'),
        (r'\bcodeigniter\b',                            'PHP Developer'),
        (r'\bsymfony\b',                                'PHP Developer'),
        (r'\bphp\b.*\b(laravel|codeigniter|symfony|artisan|blade)\b',
                                                        'PHP Developer'),
        (r'\bphp\s+developer\b',                        'PHP Developer'),
        # ── Machine Learning ────────────────────────────────────────────────
        (r'\bmlops\b',                                  'Machine Learning Engineer'),
        (r'\bml\s+pipeline\b',                          'Machine Learning Engineer'),
        (r'\bmodel\s+training\b.*\b(pytorch|tensorflow|keras)\b',
                                                        'Machine Learning Engineer'),
        (r'\b(pytorch|tensorflow)\b.*\bmodel\s+(training|deployment|pipeline)\b',
                                                        'Machine Learning Engineer'),
        (r'\bmachine\s+learning\s+engineer\b',          'Machine Learning Engineer'),
        (r'\bml\s+engineer\b',                          'Machine Learning Engineer'),
        # ── DevOps ──────────────────────────────────────────────────────────
        # Require explicit devops phrase or terraform/ansible — Kubernetes alone
        # appears in Full Stack / Software Engineer descriptions too
        (r'\bterraform\b',                              'DevOps Engineer'),
        (r'\bansible\b',                                'DevOps Engineer'),
        (r'\bci.?cd\b.*\b(jenkins|github\s+actions|gitlab\s+ci|circleci)\b',
                                                        'DevOps Engineer'),
        (r'\bhelm\b.*\bkubernetes\b',                   'DevOps Engineer'),
        (r'\bkubernetes\b.*\b(terraform|ansible|helm|prometheus|grafana)\b',
                                                        'DevOps Engineer'),
        (r'\bdocker\b.*\bkubernetes\b.*\b(jenkins|terraform|ansible|ci.?cd)\b',
                                                        'DevOps Engineer'),
        # ── Database Administrator ──────────────────────────────────────────
        (r'\bdatabase\s+administrator\b',               'Database Administrator'),
        (r'\bdba\b',                                    'Database Administrator'),
        (r'\b(oracle|sql\s+server)\b.*\b(dba|administrator|admin)\b',
                                                        'Database Administrator'),
        # ── Network Administrator ────────────────────────────────────────────
        (r'\bnetwork\s+administrator\b',                'Network Administrator'),
        (r'\b(ccna|ccnp|cisco)\b',                      'Network Administrator'),
        (r'\b(lan|wan)\b.*\b(administrator|admin|engineer)\b',
                                                        'Network Administrator'),
        # ── Java ─────────────────────────────────────────────────────────────
        (r'\bspring\s+boot\b',                          'Java Developer'),
        (r'\bhibernate\b.*\b(java|spring|maven)\b',     'Java Developer'),
        (r'\bjava\b.*\b(spring\s+boot|hibernate|maven|gradle)\b',
                                                        'Java Developer'),
    ]

    # ── Pass 2: Explicit title phrases (confidence 100) ──────────────────────
    REGEX_PATTERNS = [
        # AI / ML
        (r'\bai\s*/?\s*ml\s+engineer\b',                    'AI/ML Engineer'),
        (r'\bmachine\s+learning\s+engineer\b',               'Machine Learning Engineer'),
        (r'\bml\s+engineer\b',                               'Machine Learning Engineer'),
        (r'\bcomputer\s+vision\s+engineer\b',                'Computer Vision Engineer'),
        (r'\bcv\s+engineer\b',                               'Computer Vision Engineer'),
        (r'\bnlp\s+engineer\b',                              'NLP Engineer'),
        (r'\bnatural\s+language\s+processing\s+engineer\b',  'NLP Engineer'),
        (r'\bai\s+engineer\b',                               'AI Engineer'),
        (r'\bartificial\s+intelligence\s+engineer\b',        'AI Engineer'),
        # Software — specific before generic
        (r'\bnode[\.\s]?js\s+developer\b',                   'Node.js Developer'),
        (r'\bflutter\s+developer\b',                         'Flutter Developer'),
        (r'\bflutter\s+engineer\b',                          'Flutter Developer'),
        (r'\bios\s+developer\b',                             'iOS Developer'),
        (r'\bjava\s+developer\b',                            'Java Developer'),
        (r'\bjavascript\s+developer\b',                      'JavaScript Developer'),
        (r'\bphp\s+developer\b',                             'PHP Developer'),
        (r'\bdjango\s+developer\b',                          'Django Developer'),
        (r'\bwordpress\s+developer\b',                       'Wordpress Developer'),
        (r'\bwordpress\s+engineer\b',                        'Wordpress Developer'),
        (r'\bwp\s+developer\b',                              'Wordpress Developer'),
        (r'\bfull[\s\-]?stack\s+developer\b',                'Full Stack Developer'),
        (r'\bfull[\s\-]?stack\s+engineer\b',                 'Full Stack Developer'),
        (r'\bfullstack\s+developer\b',                       'Full Stack Developer'),
        (r'\bfrontend\s+developer\b',                        'Frontend Developer'),
        (r'\bfront[\s\-]end\s+developer\b',                  'Frontend Developer'),
        (r'\bbackend\s+developer\b',                         'Backend Developer'),
        (r'\bback[\s\-]end\s+developer\b',                   'Backend Developer'),
        (r'\bmobile\s+developer\b',                          'Mobile Developer'),
        (r'\bsoftware\s+engineer\b',                         'Software Engineer'),
        (r'\bswe\b',                                         'Software Engineer'),
        (r'\bsystems?\s+architect\b',                        'Systems Architect'),
        # DevOps / Infra
        (r'\bsite\s+reliability\s+engineer\b',               'Site Reliability Engineer'),
        (r'\bsre\b',                                         'Site Reliability Engineer'),
        (r'\bdevops\s+engineer\b',                           'DevOps Engineer'),
        (r'\bcloud\s+engineer\b',                            'Cloud Engineer'),
        (r'\bnetwork\s+administrator\b',                     'Network Administrator'),
        (r'\bnetwork\s+admin\b',                             'Network Administrator'),
        (r'\bnetwork\s+engineer\b',                          'Network Engineer'),
        (r'\bdba\b',                                         'Database Administrator'),
        (r'\bdatabase\s+administrator\b',                    'Database Administrator'),
        (r'\bcyber\s*security\s+analyst\b',                  'Cybersecurity Analyst'),
        (r'\bcybersecurity\s+analyst\b',                     'Cybersecurity Analyst'),
        (r'\binfosec\s+analyst\b',                           'Cybersecurity Analyst'),
        (r'\bsecurity\s+engineer\b',                         'Security Engineer'),
        # Data
        (r'\bdata\s+scientist\b',                            'Data Scientist'),
        (r'\bdata\s+engineer\b',                             'Data Engineer'),
        (r'\bdata\s+analyst\b',                              'Data Analyst'),
        # QA / Design
        (r'\bqa\s+engineer\b',                               'QA Engineer'),
        (r'\bquality\s+assurance\s+engineer\b',              'QA Engineer'),
        (r'\bui\s*/\s*ux\s+designer\b',                      'UI/UX Designer'),
        (r'\bux\s+designer\b',                               'UI/UX Designer'),
        # Business
        (r'\bbusiness\s+analyst\b',                          'Business Analyst'),
        (r'\bproduct\s+manager\b',                           'Product Manager'),
        (r'\bproject\s+manager\b',                           'Project Manager'),
        (r'\bscrum\s+master\b',                              'Scrum Master'),
    ]

    # ── Pass 3: Exact title keyword list ────────────────────────────────────
    KNOWN_TITLES = [
        "Machine Learning Engineer", "AI/ML Engineer",
        "Computer Vision Engineer", "NLP Engineer", "AI Engineer",
        "Node.js Developer", "Flutter Developer", "iOS Developer",
        "Java Developer", "JavaScript Developer", "PHP Developer",
        "Django Developer", "Wordpress Developer",
        "Full Stack Developer", "Frontend Developer", "Backend Developer",
        "Mobile Developer", "Software Architect", "Software Engineer",
        "Site Reliability Engineer", "DevOps Engineer", "Cloud Engineer",
        "Network Administrator", "Network Engineer", "Database Administrator",
        "Cybersecurity Analyst", "Cyber Security Analyst", "Security Engineer",
        "Systems Architect", "Data Scientist", "Data Engineer", "Data Analyst",
        "QA Engineer", "UI/UX Designer",
        "Business Analyst", "Product Manager", "Project Manager", "Scrum Master",
    ]

    def __init__(self):
        # Pre-compile all strong patterns (IGNORECASE, DOTALL for multi-line)
        self._strong = [
            (re.compile(pattern, re.IGNORECASE | re.DOTALL), title)
            for pattern, title in self.STRONG_PATTERNS
        ]
        # Pre-compile explicit title phrase patterns
        self._compiled = [
            (re.compile(pattern, re.IGNORECASE), title)
            for pattern, title in self.REGEX_PATTERNS
        ]

    def match(self, text: str):
        """
        Scan *text* for a known job title.

        Returns (canonical_title, confidence) or (None, None).

        Confidence values:
          100 — explicit title phrase found (Pass 2 / Pass 3)
           90 — strong single-signal keyword found (Pass 1)
        """
        if not text:
            return None, None

        # ── Pass 1: Strong single-signal keywords ───────────────────────────
        for compiled_pattern, canonical_title in self._strong:
            if compiled_pattern.search(text):
                return canonical_title, 90

        # ── Pass 2: Explicit title phrases ──────────────────────────────────
        for compiled_pattern, canonical_title in self._compiled:
            if compiled_pattern.search(text):
                return canonical_title, 100

        # ── Pass 3: Exact keyword word-boundary scan ─────────────────────────
        for title in self.KNOWN_TITLES:
            escaped = re.escape(title)
            if re.search(rf'\b{escaped}\b', text, re.IGNORECASE):
                return title, 100

        return None, None
