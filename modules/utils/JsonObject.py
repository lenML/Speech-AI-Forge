class JsonObject:
    def __init__(self, initial_dict=None):
        """
        Initialize the JsonObject with an optional initial dictionary.

        :param initial_dict: A dictionary to initialize the JsonObject.
        """
        # If no initial dictionary is provided, use an empty dictionary
        self._dict_obj = initial_dict if initial_dict is not None else {}

        if self._dict_obj is self:
            raise ValueError("JsonObject cannot be initialized with itself")

    def __getattr__(self, name):
        """
        Get an attribute value. If the attribute does not exist,
        look it up in the internal dictionary.

        :param name: The name of the attribute.
        :return: The value of the attribute.
        :raises AttributeError: If the attribute is not found in the dictionary.
        """
        try:
            return self._dict_obj[name]
        except KeyError:
            return None

    def __setattr__(self, name, value):
        """
        Set an attribute value. If the attribute name is '_dict_obj',
        set it directly as an instance attribute. Otherwise,
        store it in the internal dictionary.

        :param name: The name of the attribute.
        :param value: The value to set.
        """
        if name == "_dict_obj":
            super().__setattr__(name, value)
        else:
            self._dict_obj[name] = value

    def __delattr__(self, name):
        """
        Delete an attribute. If the attribute does not exist,
        look it up in the internal dictionary and remove it.

        :param name: The name of the attribute.
        :raises AttributeError: If the attribute is not found in the dictionary.
        """
        try:
            del self._dict_obj[name]
        except KeyError:
            return

    def __getitem__(self, key):
        """
        Get an item value from the internal dictionary.

        :param key: The key of the item.
        :return: The value of the item.
        :raises KeyError: If the key is not found in the dictionary.
        """
        if key not in self._dict_obj:
            return None
        return self._dict_obj[key]

    def __setitem__(self, key, value):
        """
        Set an item value in the internal dictionary.

        :param key: The key of the item.
        :param value: The value to set.
        """
        self._dict_obj[key] = value

    def __delitem__(self, key):
        """
        Delete an item from the internal dictionary.

        :param key: The key of the item.
        :raises KeyError: If the key is not found in the dictionary.
        """
        del self._dict_obj[key]

    def to_dict(self):
        """
        Convert the JsonObject back to a regular dictionary.

        :return: The internal dictionary.
        """
        return self._dict_obj

    def has_key(self, key):
        """
        Check if the key exists in the internal dictionary.

        :param key: The key to check.
        :return: True if the key exists, False otherwise.
        """
        return key in self._dict_obj

    def keys(self):
        """
        Get a list of keys in the internal dictionary.

        :return: A list of keys.
        """
        return self._dict_obj.keys()

    def values(self):
        """
        Get a list of values in the internal dictionary.

        :return: A list of values.
        """
        return self._dict_obj.values()

    def clone(self):
        """
        Clone the JsonObject.

        :return: A new JsonObject with the same internal dictionary.
        """
        return JsonObject(self._dict_obj.copy())

    def merge(self, other):
        """
        Merge the internal dictionary with another dictionary.

        :param other: The other dictionary to merge.
        """
        self._dict_obj.update(other)
