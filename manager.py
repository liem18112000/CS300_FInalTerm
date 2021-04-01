class Registry(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self._registry = {}

    def register(self, name, registee):
        if not self._registry.has_key(name):
            self._registry[name] = registee
            return True
        return False

    def get(self, name):
        if self._registry.has_key(name):
            return self._registry[name]
        return None

    def remove(self, name):
        if self._registry.has_key(name):
            del self._registry[name]
            return True
        return False


class ModelManager(object):
    def __init__(self, registry = None):
        if registry is None:
            self._registry = Registry()
        else: 
            self._registry = registry

    def registerModel(self, name, model):
        return self._registry.register(
            name = name,
            registee = model
        )

    def getModel(self, name):
        return self._registry.get(name)

    def resetRegistry(self):
        self._registry.reset()

    def removeModel(self, name):
        return self._registry.remove(name)
