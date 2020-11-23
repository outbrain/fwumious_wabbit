package com.examples;

import com.outbrain.fw.ASession;
import com.outbrain.fw.FWSession;

class Main {
    public static void main(String[] args) {
        try {
            System.loadLibrary("fw");
        } catch (UnsatisfiedLinkError e) {
            System.out.println("Can not load library");
            throw e;
        }
        System.out.println("WAAAAAAAAT");
        ASession a = new ASession();
        System.out.println(a.greet("CRAZY"));
        FWSession b = new FWSession("--data ../../examples/basic/datasets/train.vw --keep A --keep B");
        System.out.println("BAAAAAAAAT");
    }
    
}
