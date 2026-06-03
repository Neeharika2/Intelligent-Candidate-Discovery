from src.utils.company_classifier import (
    normalize_company_name,
    is_services_company,
    classify_career_services_ratio
)

def test_normalize_company_name():
    assert normalize_company_name("Google Inc.") == "google"
    assert normalize_company_name("Mindtree Limited") == "mindtree"
    assert normalize_company_name("TATA Consultancy Services Pvt. Ltd.") == "tata consultancy services"
    assert normalize_company_name("Wipro Technologies") == "wipro technologies"
    assert normalize_company_name("Acme Corp LLC") == "acme"

def test_is_services_company():
    assert is_services_company("TCS") is True
    assert is_services_company("Tata Consultancy Services") is True
    assert is_services_company("Infosys Pvt Ltd") is True
    assert is_services_company("Wipro Technologies") is True
    assert is_services_company("Accenture") is True
    assert is_services_company("Google") is False
    assert is_services_company("Dunder Mifflin") is False

def test_classify_career_services_ratio():
    # Candidate 1: Pure service career
    career_pure_service = [
        {"company": "TCS"},
        {"company": "Infosys"},
        {"company": "Wipro"}
    ]
    assert classify_career_services_ratio(career_pure_service, threshold=0.8) is True

    # Candidate 2: Mixed career (2 services, 2 products)
    career_mixed = [
        {"company": "Accenture"},
        {"company": "Cognizant"},
        {"company": "Google"},
        {"company": "Stark Industries"}
    ]
    # Unique companies: accenture, cognizant, google, stark industries (4 total, 2 service)
    # Ratio: 2/4 = 0.5. Since 0.5 < 0.8, it should be False
    assert classify_career_services_ratio(career_mixed, threshold=0.8) is False

    # Candidate 3: Edge case - empty history
    assert classify_career_services_ratio([], threshold=0.8) is False
