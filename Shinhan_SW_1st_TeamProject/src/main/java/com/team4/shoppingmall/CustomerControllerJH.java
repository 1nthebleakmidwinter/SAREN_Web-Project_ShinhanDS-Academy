package com.team4.shoppingmall;

import javax.servlet.http.HttpSession;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;

@Controller
@RequestMapping("/customer")
public class CustomerControllerJH {
	
	/*��� �޴� - ��ǰ ��� ��ȸ */
	@GetMapping("/productlist")
	public String productlist() {
		
		return "customer/productlist";
	}
	
	/*���������� ����*/
	@GetMapping("/myPage.do")
	public String myPage() {
		//1.���� �ֹ� ��� �ҷ�����
		
		//2.���� �뿩 ��� �ҷ�����
		
		return "customer/myPage";
	}

	//���� �ֹ� ����Ʈ
	@GetMapping("/orderlist")
	public String orderlist() {
		
		return "customer/orderlist";
	}
	
	//���� �뿩 ����Ʈ
	@GetMapping("/rentlist")
	public String rentlist() {
		//�뿩 ��� ��ȸ
		
		return "customer/rentlist";
	}

	/* ȸ���������� */
	//step1
	@GetMapping("/myInfoUpdate.do")
	public String myInfoUpdate() {
		
		return "customer/myInfoUpdate";
	}
	
	//step2 - ��й�ȣ Ȯ�� â
	@GetMapping("/myInfoUpdatePw.do")
	public String myInfoUpdatePw() {
		
		return "customer/myInfoUpdate_step2";
	}
	
	
	//��й�ȣ üũ �� ���� ����(step3)
	@GetMapping("/myInfoUpdatePwCheck.do")
	public String myInfoUpdatePwCheck(@RequestParam("password") String password) {
		if(password.equals("aaa")) {
			return "customer/myInfoUpdate_step3";
		}else {
			System.out.println("���������� ȸ�� ��й�ȣ Ȯ�� ����");
			return "redirect:customer/myInfoUpdate_step2";
		}
		
	}
	
	//step3 - ������ ȸ�� ���� �Է�â	
//	@PostMapping("/myInfoUpdateForm.do")
//	public String myInfoUpdateForm() {
//
//		return "";
//	}
	
	/* ȸ�� Ż�� */
	@GetMapping("/memberDelete.do")
	public String memberDelete() {
		
		return "customer/memberDelete";
	}
	//��й�ȣ üũ �� ȸ�� Ż��
	@GetMapping("/memberDeletePwCheck.do")
	public String memberDeletePwCheck(@RequestParam("password") String password) {
		if(password.equals("aaa")) {
			//System.out.println(password);
			return "customer/myPage";
		}else {
			System.out.println("ȸ�� Ż�� ����");
			return "redirect:customer/memberDelete";
		}
		
	}
	
}
