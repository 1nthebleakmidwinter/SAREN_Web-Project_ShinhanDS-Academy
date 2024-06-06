<%@ page language="java" contentType="text/html; charset=UTF-8"
	pageEncoding="UTF-8"%>
<%@taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core"%>
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>마이페이지-회원정보수정</title>
<%-- css --%>
<link rel="stylesheet" href="${path}/resources/css/mypage.css">
<%-- 헤더,푸터 css --%>
<link rel="stylesheet" href="${path}/resources/css/header_footer.css">
<%-- jquery 연결 --%>
<script src="${path}/resources/js/jquery-3.7.1.min.js"></script>
<script
	src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>
</head>
<body>
	<c:set var="path" value="${pageContext.servletContext.contextPath}" />

	<%-- 헤더 연결하기 --%>
	<%-- 마이페이지 : 회원정보수정 --%>
	<div class="mypage_wrap">
		<div class="myinfo inner">
			<nav>
				<ul>
					<li>
						<h3>
							<a href="#">나의주문</a>
						</h3>
					</li>
					<li>
						<h3>
							<a href="#">나의대여</a>
						</h3>
					</li>
					<li>
						<h3>
							<a href="#">장바구니</a>
						</h3>
					</li>
					<li>
						<h3>
							<a href="#">나의글</a>
						</h3>
						<ul class="myinfo_submenu">
							<li><a href="#">문의글</a></li>
							<li><a href="#">리뷰</a></li>
						</ul>
					</li>
					<li>
						<h3>
							<a href="#">회원정보</a>
						</h3>
						<ul class="myinfo_submenu">
							<li><a href="#">정보수정</a></li>
							<li><a href="#">회원탈퇴</a></li>
						</ul>
					</li>
				</ul>
			</nav>
			<div class="mypage_here">
				<div class="section_wrap">
					<h1 class="myinfo_title">회원 정보 수정</h1>
					<div class="section myinfo_update">
						<!-- "비밀번호 확인" 버튼 클릭: myinfo_update에서 페이지 업데이트 -->
						<div class="pw-check">
							<p>회원님의 개인정보보호를 위한 본인 확인절차를 위해 비밀번호를 입력해 주세요.</p>
							<button class="button">비밀번호 확인</button>
						</div>
					</div>
					<h2 class="myinfo_title">나의 배송지</h2>
					<div class="section">
						<div class="address-box">
							<div class="label">기본</div>
							<div class="info">
								홍길동 <strong>010-0000-0000</strong><br> [00000] 서울특별시 00구
								00로 00(00동, 00타운) A동
							</div>
							<button class="delete-btn">삭제</button>
						</div>
						<div class="adress-add">
							<button class="button">배송지 추가</button>
						</div>
					</div>
				</div>
			</div>
		</div>
	</div>
</body>
</html>