package com.examples;

import com.outbrain.fw.Session;

class Main {
    public static void main(String[] args) {
        try {
            System.loadLibrary("fw");
        } catch (UnsatisfiedLinkError e) {
            System.out.println("Can not load library");
            throw e;
        }
        System.out.println("WAAAAAAAAT");
        Session a = new Session();
    }
    
}
