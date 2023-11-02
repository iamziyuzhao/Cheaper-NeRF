package util;

import java.util.AbstractMap;
import java.util.Map;
import java.util.Set;

/**
 * A Map implementation backed by an ArraySet.
 * Good for small, more or less immutable maps, but not part of Java.
 */
public class ArrayMap<K,V> extends AbstractMap<K,V> {
	
	protected ArraySet<Map.Entry<K,V>> entries;
	
	/**
	 * Construct and return a new ArrayMap with initial capacity 0.
	 */
	public ArrayMap() {
		this(0);
	}
	
	/**
	 * Construct and return a new ArrayMap with the given initial capacity.
	 */
	public ArrayMap(int initialCapacity) {
		this.entries = new ArraySet<Map.Entry<K,V>>(initialCapacity);
	}

	/**
	 * Returns a Set view of the mappings contained in this map.
	 * Required by subclasses of AbstractMap.
	 */
	@Override
	public Set<Entry<K, V>> entrySet() {
		return this.entries;
	}

	/**
	 * Associates the specified value with the specified key in this ArrayMap.
	 * Required by AbstractMap for mutable maps. It could perhaps be
	 * implemented more efficiently...
	 */
	@Override
	public V put(K key, V value) {
		V oldValue = this.remove(key);
		Map.Entry<K,V> entry = new AbstractMap.SimpleEntry<K,V>(key, value);
		this.entries.add(entry);
		return oldValue;
	}
	
	/**
	 * Return a new ArrayMap containing the same entries as this one (so a
	 * ``shallow'' copy).
	 */
	public ArrayMap<K,V> copy() {
		ArrayMap<K,V> newMap = new ArrayMap<K,V>(this.size());
		for (Map.Entry<K,V> entry : this.entrySet()) {
			newMap.put(entry.getKey(), entry.getValue());
		}
		return newMap;
	}

}
