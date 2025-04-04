# import google.generativeai


# google.generativeai.configure(api_key="")


# models = [m for m in google.generativeai.list_models() if "generateText" in m.supported_generation_methods]
# model = models[0].name


# def palm(prompt: str) -> str:
#     completion = google.generativeai.generate_text(
#         model=model,
#         prompt=prompt
#     )
#     return completion.result
