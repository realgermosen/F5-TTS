from setuptools import setup, find_packages

setup(
    name="RealTimeTranslator",
    version="1.0.0",
    description="A real-time audio translation system using transcription, translation, and TTS.",
    author="AI Prodigy LLC",
    author_email="Support@AI-Prodigy.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "sounddevice",
        "soundfile",
        "numpy",
        "tkinter",  # Often pre-installed with Python, can be optional
        "transformers",
        "f5-tts",
        "threading",
        "tempfile",
    ],
    python_requires=">=3.10"
)
