class Colors:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"

def log_info(message:str,color:str=Colors.CYAN):
    """Log info messages with color"""
    print(f"{color}[INFO] {message}{Colors.END}") 

def log_success(message:str):
    """Log success messages with color"""
    print(f"{Colors.GREEN}[SUCCESS] {message}{Colors.END}")

def log_warning(message:str):
    """Log warning messages with color"""
    print(f"{Colors.YELLOW}[WARNING] {message}{Colors.END}")

def log_error(message:str):
    """Log error messages with color"""
    print(f"{Colors.RED}[ERROR] {message}{Colors.END}")

def log_header():
    """Log header messages with color"""
    print(f"{Colors.BOLD}{Colors.PURPLE}{"="*60}{Colors.END}")