import pydantic as pd


class ValidationResult(pd.BaseModel):
    errors: list = pd.Field([])
    warnings: list = pd.Field([])

    def passed(self):
        return len(self.errors) == 0
