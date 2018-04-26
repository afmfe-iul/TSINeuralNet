package utils;

import java.util.ArrayList;
import java.util.List;

public class DataContainer {
	private final List<Object> contents = new ArrayList<>();

	public void add(Object content) {
		contents.add(content);
	}
	
	public Object getContentAt(int index){
		return contents.get(index);
	}
}