class Auth:
    def login(self, username: str, password: str) -> bool:
        return bool(username and password)
