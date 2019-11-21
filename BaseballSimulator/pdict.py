import collections

try:
  test = unicode()
except:
  unicode = str

class pdict(collections.MutableMapping):
  """A dictionary that allows nested element access using key paths."""

  def __init__(self,*args,**kwargs):
    self.delimiter = '/'
    self.pup = '..'
    self.phere  = '.'

    self.parent = None
    self.store = dict()
    self.update(dict(*args, **kwargs))  # use the free update to set keys
    self.recursive_convert()

  def __getitem__(self, key):
    key = str(key)

    # return self for blank keys or None
    if key == "" or key is None:
      return self

    # check if path is absolute
    if key.startswith(self.delimiter):
      node = self
      while node.parent != None:
        node = node.parent
      return node[key[len(self.delimiter):]]

    # split key into path elements
    toks = key.split(self.delimiter,1)
    head = toks[0]
    tail = None if len(toks) < 2 else toks[1]

    if head == self.pup:
      # first path element references parent
      return self.parent if tail is None else self.parent[tail]
    elif head == self.phere:
      # first path element references self
      return self if tail is None else self[tail]
    else:
      # first path element references key or index
      if isinstance(self.store,list):
        # if internal storage is a list, convert head to an int
        head = int(head)
      return self.store[head] if tail is None else self.store[head][tail]

  def __setitem__(self, key, value):
    # if key is a path, we need to get the node that value will
    # be added to first. we'll create parents as needed.
    if isinstance(key,(str,unicode)):
      # strip off trailing '/' if it exists
      if key.endswith(self.delimiter):
        key = key[:-len(self.delimiter)]

      # root node is referenced
      if key.startswith(self.delimiter):
        node = self
        while node.parent != None:
          node = node.parent
        node[key[len(self.delimiter):]] = value
        return

      # split key into path elements
      toks = key.split(self.delimiter,1)
      head = toks[0]
      tail = None if len(toks) < 2 else toks[1]

      # key is a path 
      if tail is not None:
        if head not in self.store:
          self.store[head] = pdict()
          self.store[head].parent = self
        self[head][tail] = value
        return



    # if value is a dict or list, convert it to a pdict
    # this allows the user to initialize the pdict just like a regular dict
    # i.e. d = { 'val' : 10, 'range' : [ 0, 10 ] }
    if isinstance(value,(dict,list)):
      newvalue = pdict()
      newvalue.store = value
      newvalue.parent = self
      value = newvalue
    
    # if value is a pdict, set
    # its parent to this pdict and check
    # to see if any of its elements need to be converted
    # to pdicts
    if isinstance(value,pdict):
      value.parent = self
      value.recursive_convert()

    # set the value
    self.store[key] = value

  def __delitem__(self, key):
    del self.store[key]

  def __iter__(self):
    return iter(self.store)

  def __len__(self):
    return len(self.store)

  def keys(self):
    if isinstance(self.store,dict):
      return self.store.keys()
    if isinstance(self.store,list):
      return range(len(self.store))
    return None

  def path(self):
    p = []
    while self.parent is not None:
      if isinstance(self.parent.store,list):
        p.append(  str(self.parent.store.index(self)) )
      if isinstance(self.parent.store,dict):
        #     vv search for ourself in parent
        p += [k for k,v in self.parent.store.items() if v == self]
      self = self.parent
    p.append("")
    p.reverse()
    return self.delimiter.join(p)

  def pathname(self,path):
    toks = path.rsplit(self.delimiter,1)
    if len(toks) == 1:
      # only one path element, like "x"
      return ""
    if toks[0] == "":
      # if first token is empty, path was an absolute path with
      # one element. like "/grid"
      return "/"

    return toks[0]

  def basename(self,path):
    toks = path.rsplit(self.delimiter,1)
    if len(toks)>1:
      return toks[1]
    else:
      return toks[0]

  def dict(self):
    '''Return a nested dict.'''
    store = dict() if isinstance(self.store,dict) else list()
    for k in (self.store if isinstance(self.store,dict) else [i for i,v in enumerate(self.store)]):
      v = self.store[k]
      if isinstance(self.store[k], pdict):
        v = v.dict()

      if isinstance(store,list):
        store.append(v)
      else:
        store[k] = v

    return store

        
  def recursive_convert(self):
    # recursively convert any nested dict's to pdict's
    for k in (self.store if isinstance(self.store,dict) else [i for i,v in enumerate(self.store)]):
      if isinstance(self[k], (dict,list)):
        self[k] = self.store[k]

