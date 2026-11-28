"""Turkish (TR) dialogue markers, lexicons, and patterns for EQ-Layer."""

from __future__ import annotations

ACTION_MARKERS = (
    "yap",
    "ekle",
    "düzelt",
    "kontrol et",
    "bak",
    "gönder",
    "oluştur",
    "sil",
    "devam et",
    "çalıştır",
    "araştır",
    "incele",
)

STATUS_MARKERS = (
    "durum ne",
    "son durum",
    "ne kadar kaldı",
    "bitti mi",
    "oluyor mu",
    "çalışıyor mu",
)

EXPLANATION_MARKERS = (
    "neden",
    "niçin",
    "nasıl",
    "anlat",
    "açıkla",
    "ne demek",
)

VAGUE_FORMS = (
    "bu",
    "bunu",
    "şu",
    "şunu",
    "onu",
    "şey",
)

CONSTRAINT_MARKERS = (
    ("kısa", "brief"),
    ("net", "direct"),
    ("sadece", "scope_limited"),
    ("detaylı", "detailed"),
    ("tahmin etme", "no_guess"),
    ("uydurma", "no_guess"),
)

ESCALATION_MARKERS = (
    "neden",
    "sen de",
    "saçmalama",
    "yine",
    "kaç kez",
    "kaç defa",
    "anlamıyorsun",
    "sinir",
    "rezalet",
)

SOFTENERS = (
    "galiba",
    "aslında",
    "belki",
    "biraz",
    "emin değilim",
    "kısmen",
)

CORRECTION_MARKERS = (
    "hayır, onu demedim",
    "hayır onu demedim",
    "öyle değil",
    "yanlış anladın",
    "yanlış anladınız",
    "onu sormadım",
    "ben onu demedim",
    "demek istediğim",
    "kastettiğim",
)

DISCLOSURE_MARKERS = (
    "kimseye anlatma",
    "kimseye söyleme",
    "sır",
    "gizli tut",
    "benden başka kimse",
)

QUESTION_OPENERS = (
    "ne",
    "neden",
    "niçin",
    "nasıl",
    "nedir",
    "hangi",
    "kaç",
    "mı",
    "mi",
)

CONTINUE_FORMS = {
    "devam",
    "devam et",
}

EXECUTE_REFERENCE_FORMS = {
    "yap",
    "bunu yap",
    "şunu yap",
    "onu yap",
}

REVISE_REFERENCE_FORMS = {
    "düzelt",
    "bunu düzelt",
    "şunu düzelt",
    "onu düzelt",
}

ASSISTANT_OFFER_MARKERS = (
    "istersen",
    "yapabilirim",
    "yapayım",
    "yapabiliriz",
    "devam edebilirim",
    "geçebilirim",
)

USER_GOAL_MARKERS = (
    "yap",
    "ekle",
    "düzelt",
    "kontrol et",
    "bak",
    "gönder",
    "oluştur",
    "sil",
    "çalıştır",
    "araştır",
    "incele",
    "durum ne",
    "son durum",
    "neden",
    "nasıl",
    "anlat",
    "açıkla",
)

ABSOLUTE_PRIVACY_PROMISES = (
    "kimseye söylemem",
    "sadece bu sohbette kalır",
    "aramızda kalacak",
    "tamamen gizli kalacak",
)

STANCE_ADMISSIONS = (
    "haklısın",
    "sen haklısın",
    "ben yanıldım",
    "ben yanlış anladım",
    "ben hatalıydım",
)

GENERIC_PHRASES = (
    "anlıyorum",
    "haklısın, çok zor",
    "bunu anlıyorum",
)
